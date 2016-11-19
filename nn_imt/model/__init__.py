import theano
from theano import tensor
from toolz import merge

from blocks.bricks.base import application
from blocks.bricks.recurrent import recurrent
from blocks.bricks import NDimensionalSoftmax
from blocks.bricks.parallel import Parallel, Distribute

from blocks.bricks.wrappers import WithExtraDims
from blocks.utils import dict_union, dict_subset
from blocks.bricks.sequence_generators import SequenceGenerator

from blocks.bricks import (Tanh, Maxout, Linear, Logistic, FeedforwardSequence,
                           Bias, Initializable, MLP)
from blocks.bricks.attention import SequenceContentAttention, AbstractAttentionRecurrent
from blocks.bricks.base import application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import GatedRecurrent, Bidirectional
from blocks.bricks.sequence_generators import (
    LookupFeedback, Readout, SoftmaxEmitter,
    SequenceGenerator)
from blocks.utils import pack
from blocks.bricks.attention import AttentionRecurrent

from machine_translation.model import (InitializableFeedforwardSequence, LookupFeedbackWMT15, GRUInitialState)
from blocks.bricks.sequence_generators import BaseSequenceGenerator

from attention import MultipleAttentionRecurrent

from picklable_itertools.extras import equizip

theano.config.optimizer = 'None'
theano.config.traceback.limit = 20
#theano.config.exception_verbosity = 'high'

class NDimensionalLogistic(Logistic):
    decorators = [WithExtraDims()]

# WORKING: optionally return the generation probability of samples from this method
# change the expected_cost computation to take the scores as input
# this should speed up min-risk significantly, because we won't need to recompute the generation costs
# however, we still need to build the graph for all sampled inputs, so passing the cost alone won't solve the problem
class PartialSequenceGenerator(BaseSequenceGenerator):
    """
    Adds the ability to predict and sample partial target sequences by inputting both a source sequence
    and a target sequence

    This sequence generator lets us use baseline NMT models for IMT
    """

    def __init__(self, readout, transition, attentions,
                 add_contexts=True, confidence_model=None, **kwargs):
        normal_inputs = [name for name in transition.apply.sequences
                         if 'mask' not in name]
        kwargs.setdefault('fork', Fork(normal_inputs))

        if type(attentions) is not list:
            attentions = [attentions]
        self.attentions = attentions

        transition = InitialStateAttentionRecurrent(
            transition, attentions,
            add_contexts=add_contexts, name="att_trans")

        self.softmax = NDimensionalSoftmax()
        super(PartialSequenceGenerator, self).__init__(
              readout, transition, **kwargs)
        self.children.append(self.softmax)

        # (optional) include a next-word confidence model
        if confidence_model:
            self.softmax = NDimensionalSoftmax()
            self.logistic = NDimensionalLogistic()
            self.confidence_model = confidence_model

            self.children.append(self.confidence_model)
            self.children.append(self.softmax)
            self.children.append(self.logistic)


    @application
    def probs(self, readouts):
        return self.softmax.apply(readouts, extra_ndim=readouts.ndim - 2)

    @recurrent
    def generate(self, outputs, **kwargs):
        """A sequence generation step.

        Parameters
        ----------
        outputs : :class:`~tensor.TensorVariable`
            The outputs from the previous step.

        Notes
        -----
        The contexts, previous states and glimpses are expected as keyword
        arguments.

        """


        states = dict_subset(kwargs, self._state_names)
        # masks in context are optional (e.g. `attended_mask`)
        contexts = dict_subset(kwargs, self._context_names, must_have=False)
        glimpses = dict_subset(kwargs, self._glimpse_names)
        # WORKING: do we have _all_ of the glimpse names here?

        # WORKING: where there are multiple glimpses, we need to compute them all, and provide them all to the next time step
        # WORKING: this method gets called, then VariableFilter gets the outputs for beam search
        # WORKING: glimpses is a list len(glimpses) % 2 = 0
        use_additional_attention = kwargs.get('use_additional_attention', False)
        update_inputs_with_additional_attention = kwargs.pop('additional_attention_in_internal_states', True)

        next_glimpses = self.transition.take_glimpses(
            as_dict=True, use_additional_attention=use_additional_attention, **dict_union(states, glimpses, contexts))
        next_readouts = self.readout.readout(
            feedback=self.readout.feedback(outputs),
            **dict_union(states, next_glimpses, contexts))
        next_outputs = self.readout.emit(next_readouts)
        next_costs = self.readout.cost(next_readouts, next_outputs)
        next_feedback = self.readout.feedback(next_outputs)
        next_inputs = (self.fork.apply(next_feedback, as_dict=True)
                       if self.fork else {'feedback': next_feedback})
        next_states = self.transition.compute_states(
            as_list=True,
            additional_attention_in_internal_states=update_inputs_with_additional_attention,
            **dict_union(next_inputs, states, next_glimpses, contexts))

        # TODO: switch to directly getting the probs from softmax
        next_probs = self.softmax.apply(next_readouts)

        # WORKING: switch from merged_states to final_states
        # also query the confidence model
        #next_merged_states = self.readout.merged_states(
        #    feedback=self.readout.feedback(outputs),
        #    **dict_union(states, next_glimpses, contexts))

        next_confidences = self.confidence_model.apply(next_states[0])

        # Note: we need to always get outputs and costs at the same index, regarless how many glimpses there are
        return (next_states + [next_outputs] + 
                list(next_glimpses.values()) +  [next_probs] + [next_confidences] + [next_costs])

    @generate.delegate
    def generate_delegate(self):
        return self.transition.apply

    @generate.property('states')
    def generate_states(self):
        return self._state_names + ['outputs'] + self._glimpse_names

    @generate.property('outputs')
    def generate_outputs(self):
        return (self._state_names + ['outputs'] +
                self._glimpse_names + ['costs'] + ['word_probs'] + ['word_confidences'])

    def get_dim(self, name):
        if name in (self._state_names + self._context_names +
                        self._glimpse_names):
            return self.transition.get_dim(name)
        elif name == 'outputs':
            return self.readout.get_dim(name)
        return super(BaseSequenceGenerator, self).get_dim(name)

    # Note: this function is only used by self.generate, because that function has the recurrent decorator
    # Note: it's not used in the cost computation
    @application
    def initial_states(self, batch_size, *args, **kwargs):
        """
        Use the kwargs to drive the initial states -- if user passed a prefix, then compute all of the states for
        that prefix, then set the initial states for the generator as the final
        states of the prefix
        """

        if 'target_prefix' in kwargs:
            # Note the transpose
            target_prefix = kwargs['target_prefix'].T
            # TODO: in the batch implementation, each target prefix will have different lengths,
            # TODO: what is the right way to deal with this? -- what are usecases where user would want to
            # TODO: pass prefixes of different lengths, or is this only relevant at training time?
            # TODO: let user pass mask -- get the actual final states using the mask
            # TODO: how is the attended mask handled at prediction time -- does this make a difference?
            mask = None

            # Prepare input for the iterative part
            states = dict_subset(kwargs, self._state_names, must_have=False)

            # masks in context are optional (e.g. `attended_mask`)

            contexts = dict_subset(kwargs, self._context_names, must_have=False)
            prefix_context_names = ['attended', 'attended_mask']
            prefix_contexts = dict_subset(contexts, prefix_context_names, must_have=True)

            # WORKING: does feedback need to be rolled one forward here?
            feedback = self.readout.feedback(target_prefix)
            inputs = self.fork.apply(feedback, as_dict=True)

            # Run the recurrent network
            results = self.transition.apply(
                mask=mask, return_initial_states=True, as_dict=True,
                **dict_union(inputs, states, prefix_contexts))

            # Remember, glimpses are computed _before_ output stage, states are
            # computed after.
            states = {name: results[name] for name in self._state_names}
            # this is just to keep the 'batch_size' parameter in the graph, because it's used by beam search in blocks
            states['states'] = states['states'].reshape((states['states'].shape[0],
                                                         batch_size, states['states'].shape[2]))

            glimpses = {name: results[name] for name in self._glimpse_names}
            glimpse_dict = {k: v[-1] for k,v in glimpses.items()}
            # WORKING: we need to reset these to the correct values
            # WORKING: issue -- the prefix initial state is a dummy, but the states of this transition were initialized
            # WORKING: with the dimensions of the source, therefore those are wrong
            # WORKING: solution: replace those states with the prefix dims
            if 'weights_0' in glimpse_dict:
                prefix_representation = kwargs['prefix_representation']
                glimpse_dict['weighted_averages_0'] = tensor.zeros((prefix_representation.shape[1], prefix_representation.shape[-1]))
                glimpse_dict['weights_0'] = tensor.zeros((prefix_representation.shape[1], prefix_representation.shape[0]))


            # the initial states of the default sequence generator are:
            # ['states', 'outputs', 'weighted_averages', 'weights'] (the last two are in the glimpses)
            state_dict = dict_union({
                'states': states['states'][-1],
                'outputs':  target_prefix[-1],
            }, glimpse_dict)

        else:
            state_dict = dict(
                self.transition.initial_states(
                    batch_size, as_dict=True, *args, **kwargs),
                    outputs=self.readout.initial_outputs(batch_size))

        # make a list of the initial states in the order required by self.generate
        return [state_dict[state_name]
                for state_name in self.generate.states]

    @initial_states.property('outputs')
    def initial_states_outputs(self):
        return self.generate.states

    @application
    def compute_states(self, outputs, mask=None, **kwargs):
        """
        Return the result of running the recurrent transition over the provided outputs (which are transformed to inputs
        via `self.readout.feedback`

        :param outputs:
        :param mask:
        :param kwargs:
        :return:

        # WORKING: move this
        Note that this is just one way that a representation of the prefix can be computed. One alternative would be
        to create a bidirectional over the prefix

        """

        # TODO: support initial_states and initial_glimpses in this method

        # run the model through the target prefix, then init the model with the correct states
        feedback = self.readout.feedback(outputs)
        prefix_inputs = self.fork.apply(feedback, as_dict=True)

        # Prepare input for the iterative part
        states = dict_subset(kwargs, self._state_names, must_have=False)

        # masks in context are optional (e.g. `attended_mask`)
        contexts = dict_subset(kwargs, self._context_names, must_have=False)

        # first run the recurrent transition for the target_prefix, then use the final states from the
        # the prefix to initialize the suffix generation
        results = self.transition.apply(
            mask=mask, return_initial_states=True, as_dict=True,
            **dict_union(prefix_inputs, states, contexts))

        return results

        # TODO: add the target prefix as context -- will this speed up training?
        # @property
        # def _context_names(self):
        #     existing_contexts = super(PartialSequenceGenerator, self)._context_names
        #     return existing_contexts + ['target_prefix']

    @application
    def cost_matrix(self, application_call, outputs, prefix_outputs, mask=None, prefix_mask=None, **kwargs):
        """Returns word-level cross-entropy generation costs for output sequences, conditioned
        upon both the source sequence, and a target prefix

        This method includes the ability to scale word-level cost by position in the sequence

        First word or first-N word training can be enabled by changing the mask that is used for the cost computation

        See Also
        --------
        :meth:`cost` : Scalar cost.

        """
        # We assume the data has axes (time, batch, features, ...)
        batch_size = outputs.shape[1]

        # TODO: move all of the state computation logic to self.compute_states, use kwargs to support initial states and glimpses
        # run the model through the target prefix, then init the model with the correct states
        prefix_feedback = self.readout.feedback(prefix_outputs)

        states = dict_subset(kwargs, self._state_names, must_have=False)

        # masks in context are optional (e.g. `attended_mask`)
        contexts = dict_subset(kwargs, self._context_names, must_have=False)

        # NOTE: we currently need to know beforehand which contexts are needed for the prefix transition, and which are needed for the decoder transition
        prefix_context_names = ['attended', 'attended_mask']
        prefix_contexts = dict_subset(contexts, prefix_context_names, must_have=True)

        prefix_results = self.compute_states(prefix_outputs, mask=prefix_mask, **dict_union(states, prefix_contexts))

        prefix_initial_states = [prefix_results[name][-1] for name in self._state_names]

        # we need the initial glimpses for every attention brick
        # we can only init the initial glimpses for the prefix from the previous transition,
        # the initial glimpses for the additional attentions must be initialized from scratch (or use the prefix initial glimpse?)
        # NOTE: remember that the previous glimpses aren't actually used in our model anyway
        # TODO: does this make sense for the initial glimpses? these are the glimpses we used
        # TODO: to compute the last word of the prefix
        # TODO: We can only get the initial glimpses for the original attention, not for all attentions
        prefix_initial_glimpses = [prefix_results[name][-1] for name in self._glimpse_names]

        # Now compute the suffix representation, and use the prefix initial states to init the recurrent transition
        feedback = self.readout.feedback(outputs)
        inputs = self.fork.apply(feedback, as_dict=True)

        # Run the recurrent network
        # this creates a second transition which includes attention over the prefix states
        # this transition can only be trained ("tuned" for IMT) -- baseline is the transition without multiple attentions
        # (some of) the pre-trained parameters can optionally be held static
        results = self.transition.apply(
            mask=mask, return_initial_states=True, as_dict=True,
            initial_states=prefix_initial_states,
            initial_glimpses=prefix_initial_glimpses,
            additional_attention_in_internal_states=kwargs.get('additional_attention_in_internal_states', True),
            **dict_union(inputs, states, contexts))

        # Separate the deliverables. The last states are discarded: they
        # are not used to predict any output symbol. The initial glimpses
        # are discarded because they are not used for prediction.
        # Remember, glimpses are computed _before_ output stage, states are
        # computed after.
        states = {name: results[name][:-1] for name in self._state_names}
        glimpses = {name: results[name][1:] for name in self._glimpse_names}

        # Compute the cost
        # Note: setting the first element of feedback to the last feedback of the prefix

        # WORKING: fix errors with feedback
        feedback = tensor.roll(feedback, 1, 0)
        # note that we subtract 1 from the summed mask to get the correct index
        feedback = tensor.set_subtensor(feedback[0],
               prefix_feedback[prefix_mask.sum(axis=0).astype('int16')-1, tensor.arange(batch_size), :])
        # WORKING: is this line why we can't learn??
        # WORKING: We need to set the feedback to the last _real_ input of the prefix -- ie sum mask to get the real lengths and select the final feedback
        # feedback = tensor.set_subtensor(feedback[0],
        #         self.readout.feedback(self.readout.initial_outputs(batch_size)))

        readouts = self.readout.readout(
            feedback=feedback, **dict_union(states, glimpses, contexts))
        costs = self.readout.cost(readouts, outputs)

        # WORKING: scale costs by position 
        # TODO: make this optional
        # idea: tile an arange to match the shape of costs, then scale by reciprocal of position
        # idx_range = tensor.arange(1, costs.shape[-1] + 1)
        # position_coeffs = 1. / tensor.tile(idx_range, [costs.shape[0], 1])

        # scale costs by word position
        # costs *= position_coeffs
        # WORKING: END scale costs by position

        if mask is not None:
            costs *= mask

        for name, variable in list(glimpses.items()) + list(states.items()):
            application_call.add_auxiliary_variable(
                variable.copy(), name=name)

        # This variables can be used to initialize the initial states of the
        # next batch using the last states of the current batch.
        # Chris: why/where/how would this be done?
        for name in self._state_names + self._glimpse_names:
            application_call.add_auxiliary_variable(
                results[name][-1].copy(), name=name+"_final_value")

        return costs

    # WORKING: return the final states -- intuitively the last thing that happens at this timestep
    def get_final_states(self, outputs, prefix_outputs, mask=None, prefix_mask=None, **kwargs):
        """Returns the final states at this timestep -- compute the glimpses, readouts and emissions, then run the transition
         to get the states. The state representation is computed _after_ a word has been predicted

        """

        # We assume the data has axes (time, batch, features, ...)
        batch_size = outputs.shape[1]

        # run the model through the target prefix, then init the model with the correct states
        prefix_feedback = self.readout.feedback(prefix_outputs)
        prefix_inputs = self.fork.apply(prefix_feedback, as_dict=True)

        # Prepare input for the iterative part
        states = dict_subset(kwargs, self._state_names, must_have=False)

        # masks in context are optional (e.g. `attended_mask`)
        contexts = dict_subset(kwargs, self._context_names, must_have=False)

        # first run the recurrent transition for the target_prefix, then use the final states from the
        # the prefix to initialize the suffix generation
        prefix_results = self.transition.apply(
            mask=prefix_mask, return_initial_states=True, as_dict=True,
            **dict_union(prefix_inputs, states, contexts))

        # TODO: does this make sense for the initial glimpses? these are the glimpses we used
        # TODO: to compute the last word of the prefix
        prefix_initial_states = [prefix_results[name][-1] for name in self._state_names]

        prefix_initial_glimpses = [prefix_results[name][-1] for name in self._glimpse_names]

        # Now compute the suffix representation, and use the prefix initial states to init the recurrent transition
        feedback = self.readout.feedback(outputs)
        inputs = self.fork.apply(feedback, as_dict=True)

        # Run the recurrent network
        results = self.transition.apply(
            mask=mask, return_initial_states=True, as_dict=True,
            initial_states=prefix_initial_states,
            initial_glimpses=prefix_initial_glimpses,
            **dict_union(inputs, states, contexts))

        # the initial state is not used because it's generated before we've predicted anything
        states = {name: results[name][1:] for name in self._state_names}

        return states['states']

    def get_readouts(self, outputs, prefix_outputs, mask=None, prefix_mask=None, **kwargs):
        """Returns the readouts (before post-merge) for every timestep in time-major matrices

        """

        # We assume the data has axes (time, batch, features, ...)
        batch_size = outputs.shape[1]

        # run the model through the target prefix, then init the model with the correct states
        prefix_feedback = self.readout.feedback(prefix_outputs)
        prefix_inputs = self.fork.apply(prefix_feedback, as_dict=True)

        # Prepare input for the iterative part
        states = dict_subset(kwargs, self._state_names, must_have=False)

        # masks in context are optional (e.g. `attended_mask`)
        contexts = dict_subset(kwargs, self._context_names, must_have=False)

        # first run the recurrent transition for the target_prefix, then use the final states from the
        # the prefix to initialize the suffix generation
        prefix_results = self.transition.apply(
            mask=prefix_mask, return_initial_states=True, as_dict=True,
            **dict_union(prefix_inputs, states, contexts))

        # TODO: does this make sense for the initial glimpses? these are the glimpses we used
        # TODO: to compute the last word of the prefix
        prefix_initial_states = [prefix_results[name][-1] for name in self._state_names]

        prefix_initial_glimpses = [prefix_results[name][-1] for name in self._glimpse_names]

        # Now compute the suffix representation, and use the prefix initial states to init the recurrent transition
        feedback = self.readout.feedback(outputs)
        inputs = self.fork.apply(feedback, as_dict=True)

        # Run the recurrent network
        results = self.transition.apply(
            mask=mask, return_initial_states=True, as_dict=True,
            initial_states=prefix_initial_states,
            initial_glimpses=prefix_initial_glimpses,
            **dict_union(inputs, states, contexts))

        # Separate the deliverables. The last states are discarded: they
        # are not used to predict any output symbol. The initial glimpses
        # are discarded because they are not used for prediction.
        # Remember, glimpses are computed _before_ output stage, states are
        # computed after.
        states = {name: results[name][:-1] for name in self._state_names}
        glimpses = {name: results[name][1:] for name in self._glimpse_names}

        # Compute the cost
        # Note: setting the first element of feedback to the last feedback of the prefix
        feedback = tensor.roll(feedback, 1, 0)
        feedback = tensor.set_subtensor(feedback[0], prefix_feedback[-1])

        readouts = self.readout.readout(
            feedback=feedback, **dict_union(states, glimpses, contexts))

        return readouts

    @application(outputs=['merged_states'])
    def get_merged_states(self, outputs, prefix_outputs, mask=None, prefix_mask=None, **kwargs):
        """Returns the readouts (before post-merge) for every timestep in time-major matrices

        """
        # We assume the data has axes (time, batch, features, ...)
        batch_size = outputs.shape[1]

        # run the model through the target prefix, then init the model with the correct states
        prefix_feedback = self.readout.feedback(prefix_outputs)
        prefix_inputs = self.fork.apply(prefix_feedback, as_dict=True)

        # Prepare input for the iterative part
        states = dict_subset(kwargs, self._state_names, must_have=False)

        # masks in context are optional (e.g. `attended_mask`)
        contexts = dict_subset(kwargs, self._context_names, must_have=False)

        # first run the recurrent transition for the target_prefix, then use the final states from the
        # the prefix to initialize the suffix generation
        prefix_results = self.transition.apply(
            mask=prefix_mask, return_initial_states=True, as_dict=True,
            **dict_union(prefix_inputs, states, contexts))

        # TODO: does this make sense for the initial glimpses? these are the glimpses we used
        # TODO: to compute the last word of the prefix
        prefix_initial_states = [prefix_results[name][-1] for name in self._state_names]

        prefix_initial_glimpses = [prefix_results[name][-1] for name in self._glimpse_names]

        # Now compute the suffix representation, and use the prefix initial states to init the recurrent transition
        feedback = self.readout.feedback(outputs)
        inputs = self.fork.apply(feedback, as_dict=True)

        # Run the recurrent network
        results = self.transition.apply(
            mask=mask, return_initial_states=True, as_dict=True,
            initial_states=prefix_initial_states,
            initial_glimpses=prefix_initial_glimpses,
            **dict_union(inputs, states, contexts))

        # Separate the deliverables. The last states are discarded: they
        # are not used to predict any output symbol. The initial glimpses
        # are discarded because they are not used for prediction.
        # Remember, glimpses are computed _before_ output stage, states are
        # computed after.
        states = {name: results[name][:-1] for name in self._state_names}
        glimpses = {name: results[name][1:] for name in self._glimpse_names}

        # Compute the cost
        # Note: setting the first element of feedback to the last feedback of the prefix
        feedback = tensor.roll(feedback, 1, 0)
        feedback = tensor.set_subtensor(feedback[0], prefix_feedback[-1])

        merged_states = self.readout.merged_states(
            feedback=feedback, **dict_union(states, glimpses, contexts))

        return merged_states

    @application(outputs=['confidences'])
    def confidence_predictions(self, application_call, readouts):

        #merged_states = merged_states.reshape((time_major_shape[0]*time_major_shape[1], time_major_shape[2]))
        # confidence = theano.tensor.nnet.sigmoid(self.confidence_model.apply(readouts))
        confidence = self.confidence_model.apply(readouts)

        # the last dimension size = 1, so we can do this
        confidences = confidence.reshape(confidence.shape[:2])

        return confidences

    # get confidence model outputs and compare to the references from the datastream, compute binary CE as cost
    @application
    def confidence_cost(self, application_call, outputs, prefix_outputs, prediction_tags, merged_states, mask=None, prefix_mask=None, **kwargs):

        confidence = self.confidence_predictions(merged_states)

        # the last dimension size = 1, so we can do this
        confidence = confidence.reshape(confidence.shape[:2])

        #confidence = confidence.reshape(time_major_shape[:2])

        # confidence = confidence.reshape(time_major_shape[:2])
        confidence_cost = theano.tensor.nnet.binary_crossentropy(confidence, prediction_tags)

        # confidence_cost should be (time, batch)
        if mask is not None:
           confidence_cost *= mask

        return confidence_cost

    @application
    def prediction_tags(self, application_call, outputs, prefix_outputs, mask=None, prefix_mask=None, **kwargs):
        """
        returns the cost of predicting the next word, which is equivalent to the negative log probability that the next
        word is correct (the next word corresponds to the first word of the suffix, aka the first word that the model
        would generate
        """

        readouts = self.get_readouts(outputs, prefix_outputs, mask, prefix_mask, **kwargs)

        # WORKING: use the states _after_ the output token has been generated
        # merged_states = self.get_merged_states(outputs, prefix_outputs, mask, prefix_mask, **kwargs)
        merged_states = self.get_final_states(outputs, prefix_outputs, mask, prefix_mask, **kwargs)

        # get the model emissions at every timestep
        # y_emissions = readouts.argmax(axis=0)
        y_emissions = readouts.argmax(axis=-1)
        y_equal = y_emissions - outputs

        # if they're not zero, they're wrong
        wrong_idxs_r = (y_equal > 0.).nonzero()
        wrong_idxs_l = (y_equal < 0.).nonzero()

        y_true = theano.tensor.ones(y_equal.shape, dtype='float32')
        y_true = theano.tensor.set_subtensor(y_true[wrong_idxs_r], 0.)
        y_true = theano.tensor.set_subtensor(y_true[wrong_idxs_l], 0.)

        # WORKING: here we wish to get only the first element of each readout,
        # WORKING: then we'll compare that to the reference to compute the true_y
        # WORKING: we'll then consult our confidence_model to get a prediction about the next y, and compare with true_y
        # WORKING: to get the cost
        # WORKING: start with just linear transformation of the output logits

        # WORKING: actually start by getting the confidence at every timestep
        # readouts should be time,batch,vocab -- Note: another merge brick could be used in the same way as the Readout
        # Note: brick to do a different kind of transformation on (states, glimpses, contexts)

        # TODO: consider reshaping for clarity -- there is an error here somewhere
        # TODO: make sure confidence model will work with 3d input
        #readout_shape = readouts.shape
        #flat_readouts = readouts.reshape((readout_shape[0]*readout_shape[1], readout_shape[2]))
        #confidence = self.confidence_model.apply(flat_readouts)
        #confidence_logits = theano.tensor.nnet.sigmoid(confidence)

        # WORKING here
        #confidence_cost = confidence_logits
        #confidence_cost = confidence_cost.reshape((readout_shape[0], readout_shape[1], 1))

        # TODO: dump everything in this function and look at it -- remember attaching auxilliary variables to the application call

        #flat_y = y_true.reshape((readout_shape[0]*readout_shape[1], 1))



        # TODO: hang auxiliary variables and monitor them to see what their shapes are


        # TODO: why the dimshuffle here? this could be a problem with the data transposes
        # confidence = confidence.dimshuffle(2,0,1)

        #confidence_cost = theano.tensor.nnet.binary_crossentropy(confidence, y_true)

        # TODO: just the first timestep?
        # confidence_cost = theano.tensor.set_subtensor(confidence_cost[1:], 0.)

        # see here for more advice on avoiding nans
        # T.clip(x, 1e-7, 1.0 - 1e-7)
        # https: // groups.google.com / forum /  # !topic/theano-users/tn0ang57mfE


        # TODO: separate confidence score computation and cost computation into different functions so that we can
        # TODO: also return the score at inference time

        # confidence should be (time, batch)
        #if mask is not None:
        #    confidence_cost *= mask
        #
        # return confidence_cost

        #application_call.add_auxiliary_variable(confidence, name='confidence')
        #application_call.add_auxiliary_variable(y_true, name='y_true')
        #application_call.add_auxiliary_variable(confidence_cost, name='confidence_cost')
        # application_call.add_auxiliary_variable(confidence)

        return readouts, merged_states


# TODO: change the interface of this sequence generator to compute costs at sampling time, not cost time
# TODO: this should make the min-risk sampling considerably faster
class MinRiskPartialSequenceGenerator(PartialSequenceGenerator):


    def __init__(self, *args, **kwargs):
        super(MinRiskPartialSequenceGenerator, self).__init__(*args, **kwargs)


    # TODO: check where 'target_samples_mask' is used -- do we need a mask for context features (probably not)
    # Note: the @application decorator inspects the arguments, and transparently adds args  ('application_call')
    @application(inputs=['representation', 'source_sentence_mask',
                         'target_samples_mask', 'target_samples', 'scores'],
                 outputs=['cost'])
    def expected_cost(self, application_call, representation, source_sentence_mask,
                      target_samples, target_samples_mask, scores, smoothing_constant=0.005,
                      **kwargs):
        """
        emulate the process in sequence_generator.cost_matrix, but compute log probabilities instead of costs
        for each sample, we need its probability according to the model (these could actually be passed from the
        sampling model, which could be more efficient)
        """

        # Transpose everything (note we can use transpose here only if it's 2d, otherwise we need dimshuffle)
        source_sentence_mask = source_sentence_mask.T

        # make samples (time, batch)
        samples = target_samples.T
        samples_mask = target_samples_mask.T

        # WORKING HERE: do the prefix representation computation
        # run the model through the target prefix, then init the model with the correct states
        # add the initial state context features
        # TODO: add 'prefix_outputs' and 'prefix_mask' to this function's clients
        # TODO: testing the transpose here -- is this the problem??
        prefix_outputs = kwargs['prefix_outputs'].T
        prefix_mask = kwargs['prefix_mask'].T

        # we need this to set the 'attended' kwarg
        keywords = {
            'mask': target_samples_mask,
            'outputs': target_samples,
            'attended': representation,
            'attended_mask': source_sentence_mask
        }

        batch_size = samples.shape[1]

        prefix_feedback = self.readout.feedback(prefix_outputs)
        prefix_inputs = self.fork.apply(prefix_feedback, as_dict=True)

        # Prepare input for the iterative part
        # TODO: we're not passing any state names, how are they actually used?
        states = dict_subset(keywords, self._state_names, must_have=False)

        # masks in context are optional (e.g. `attended_mask`)
        contexts = dict_subset(keywords, self._context_names, must_have=False)

        # first run the recurrent transition for the target_prefix, then use the final states from the
        # the prefix to initialize the suffix generation
        prefix_results = self.transition.apply(
            mask=prefix_mask, return_initial_states=True, as_dict=True,
            **dict_union(prefix_inputs, states, contexts))

        # TODO: does this make sense for the initial glimpses? these are the glimpses we used
        # TODO: to compute the last word of the prefix
        prefix_initial_states = [prefix_results[name][-1] for name in self._state_names]

        prefix_initial_glimpses = [prefix_results[name][-1] for name in self._glimpse_names]

        # WORKING: get the initial glimpses for all additional attentions

        # Now compute the suffix representation, and use the prefix initial states to init the recurrent transition
        feedback = self.readout.feedback(samples)
        inputs = self.fork.apply(feedback, as_dict=True)

        # Run the recurrent network
        results = self.transition.apply(
            mask=samples_mask, return_initial_states=True, as_dict=True,
            initial_states=prefix_initial_states,
            initial_glimpses=prefix_initial_glimpses,
            **dict_union(inputs, states, contexts))

        # Separate the deliverables. The last states are discarded: they
        # are not used to predict any output symbol. The initial glimpses
        # are discarded because they are not used for prediction.
        # Remember, glimpses are computed _before_ output stage, states are
        # computed after.
        states = {name: results[name][:-1] for name in self._state_names}
        glimpses = {name: results[name][1:] for name in self._glimpse_names}

        # Compute the cost
        # Note: setting the first element of feedback to the last feedback of the prefix
        feedback = tensor.roll(feedback, 1, 0)
        feedback = tensor.set_subtensor(feedback[0], prefix_feedback[-1])
        readouts = self.readout.readout(
            feedback=feedback, **dict_union(states, glimpses, contexts))

        # WORKING: use the seq_probs passed in from the sampler --
        # TODO: we can't do that because theano can't infer the graph
        # TODO: find another way to get the sequence probs at sample time
        # TODO: the only obvious way is to sample inside this function, but that means we can't do any post processing on the samples

        word_probs = self.probs(readouts)
        word_probs = tensor.log(word_probs)

        # Note: converting the samples to one-hot wastes space, but it gets the job done
        # a different way of getting the one hots
        flat_samples = samples.flatten()
        zeros = theano.tensor.zeros((flat_samples.shape[0], word_probs.shape[-1]))
        #arange for first 2 dims, then flat samples
        one_hot_samples = theano.tensor.set_subtensor(zeros[theano.tensor.arange(flat_samples.shape[0]), flat_samples], 1.)
        one_hot_samples = one_hot_samples.reshape(word_probs.shape)

        one_hot_samples.astype('float32')
        actual_probs = word_probs * one_hot_samples

        # reshape to (batch, time, prob), then sum over the batch dimension
        # to get sequence-level probability
        actual_probs = actual_probs.dimshuffle(1,0,2)
        # we are first summing over vocabulary (only one non-zero cell per row), just reduces dim by 1
        sequence_probs = actual_probs.sum(axis=2)
        sequence_probs = sequence_probs * target_samples_mask

        # TODO: normalize sequence probs??
        # now sum over time dimension (because we're in log domain)
        sequence_probs = sequence_probs.sum(axis=1)

        # reshape and do exp() to get the true probs back
        sequence_probs = sequence_probs.reshape(scores.shape)

        # Note that the smoothing constant can be set by user
        sequence_distributions = (tensor.exp(sequence_probs*smoothing_constant) /
                                  tensor.exp(sequence_probs*smoothing_constant)
                                  .sum(axis=1, keepdims=True))

        # the following lines are done explicitly for code clarity
        # -- first get sequence expectation, then sum up the expectations for every
        # seq in the minibatch
        expected_scores = (sequence_distributions * scores).sum(axis=1)
        expected_scores = expected_scores.sum(axis=0)

        return expected_scores

class InitialStateAttentionRecurrent(MultipleAttentionRecurrent):
    """
    Allow user-specified initial states in the recurrent transition
    """

    def __init__(self, *args, **kwargs):
        super(InitialStateAttentionRecurrent, self).__init__(*args, **kwargs)

    @application
    def initial_states(self, batch_size, **kwargs):
        """
        Allow user to either pass initial states, or pass through to default behavior
        """
        # WORKING: initial_glimpses can be a list
        if 'initial_states' in kwargs and 'initial_glimpses' in kwargs:
            # Note: these states currently get popped out of AttentionRecurrent kwargs in the same way as mmmt
            # Note: the modification is in MultipleAttentionRecurrent.compute_states
            # WORKING: get initial states for each of the attentions, make sure we can reuse the params for:
            # WORKING: (1) the first pass over the prefix
            # WORKING: (2) the second pass which uses the prefix to initialize, and additionally has attention over the prefix
            transition_initial_states = kwargs.pop('initial_states')
            attention_initial_glimpses = kwargs.pop('initial_glimpses')
            # HACK -- we have to cut off the last two, because those were dummies from the first time we ran the transition
            attention_initial_glimpses = attention_initial_glimpses[:2]

            additional_attention_initial_glimpses = []
            for additional_attention, additional_attended_name in zip(self.additional_attentions,
                                                                      self.additional_attended_names):

                # Note the name shortening hack because of the hardcoded name on the Attention brick `get_dim`
                more_initial_glimpses = additional_attention.initial_glimpses(batch_size, kwargs[additional_attended_name])
                additional_attention_initial_glimpses.extend(pack(more_initial_glimpses))

            initial_states = (pack(transition_initial_states) + pack(attention_initial_glimpses) +
                              pack(additional_attention_initial_glimpses))
        else:
            initial_states = super(InitialStateAttentionRecurrent, self).initial_states(batch_size, **kwargs)

        # when special initial states aren't available, and we're not using the additional attention -- i.e. when computing the representation for the prefix, we need to return dummy initial states
        return initial_states

    @initial_states.property('outputs')
    def initial_states_outputs(self):
        return self.do_apply.states


# TODO: make the prefix representation configurable, so that we can swap out modules (forward recurrent, bidir, attention)
# TODO: make sure the prefix mask is handled correctly
class NMTPrefixDecoder(Initializable):
    """
    This decoder lets you use a trained NMT model for IMT prediction without changing anything

    Parameters:
    -----------
    vocab_size: int
    embedding_dim: int
    representation_dim: int
    theano_seed: int
    loss_function: str : {'cross_entropy'(default) | 'min_risk'}

    """

    def __init__(self, vocab_size, embedding_dim, state_dim,
                 representation_dim, theano_seed=None, loss_function='cross_entropy',
                 use_post_merge=True,
                 prefix_attention=False,
                 prefix_attention_in_readout=False,
                 **kwargs):

        super(NMTPrefixDecoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.representation_dim = representation_dim
        self.theano_seed = theano_seed

        # Initialize gru with special initial state
        self.transition = GRUInitialState(
            attended_dim=state_dim, dim=state_dim,
            activation=Tanh(), name='decoder')

        # Initialize the attention mechanism(s)
        # WORKING: attentions go in a list, but the first one falls back to NIMT default behavior
        #  the prefix attention is only used in the _second_ recurrent transition of the decoder
        # this means that additional attentions need to be optional each time transition.apply is called
        if prefix_attention:
            self.attention = []
            self.attention.append(SequenceContentAttention(
                state_names=self.transition.apply.states,
                attended_dim=representation_dim,
                match_dim=state_dim, name="attention"))

            # Note the name for the additional attention
            additional_attention = SequenceContentAttention(
                state_names=self.transition.apply.states,
                attended_dim=representation_dim,
                match_dim=state_dim, name="prefix_attention")

            self.attention.append(additional_attention)

        else:
            self.attention = SequenceContentAttention(
                state_names=self.transition.apply.states,
                attended_dim=representation_dim,
                match_dim=state_dim, name="attention")


        # (optional) add the confidence model
        confidence_model = InitializableFeedforwardSequence([
                 # Linear(input_dim=vocab_size, output_dim=1000,
                 #        use_bias=True, name='confidence_model0').apply,
                 # Bias(dim=state_dim, name='maxout_bias').apply,
                 # Maxout(num_pieces=2, name='maxout').apply,
                 # Linear(input_dim=state_dim / 2, output_dim=300, use_bias=True, name='confidence_model1').apply,

                 # adding the softmax feature is currently hacked because it's not supported at inference time (in search and generation)
                 # we add one to the state dim because we added the softmax argmax probability as a feature
                 # Linear(input_dim=state_dim + 1, output_dim=300, use_bias=True, name='confidence_model1').apply,
                 Linear(input_dim=state_dim, output_dim=300, use_bias=True, name='confidence_model1').apply,
                 Tanh().apply,
                 Linear(input_dim=300, output_dim=100, use_bias=True, name='confidence_model2').apply,
                 Tanh().apply,
                 Linear(input_dim=100, output_dim=1, use_bias=True, name='confidence_model3').apply,
                 Logistic().apply])
                 # Linear(input_dim=state_dim, output_dim=1, use_bias=True, name='confidence_model1').apply])


        # we allow use post merge to be configurable so that we can train confidence models which directly use
        # the output of the Readout brick
        if use_post_merge:
            readout_post_merge = InitializableFeedforwardSequence(
                [Bias(dim=state_dim, name='maxout_bias').apply,
                 Maxout(num_pieces=2, name='maxout').apply,
                 Linear(input_dim=state_dim / 2, output_dim=embedding_dim,
                        use_bias=False, name='softmax0').apply,
                 Linear(input_dim=embedding_dim, name='softmax1').apply])
        else:
            readout_post_merge = None

        # Initialize the readout, note that SoftmaxEmitter emits -1 for
        # initial outputs which is used by LookupFeedBackWMT15

        # Chris: it's key that we're taking the first output of self.attention.take_glimpses.outputs
        # Chris: the first output is the weighted avgs, the second is the weights in (batch, time)
        if type(self.attention) is list:
            attention_sources = []
            attention_sources.append(self.attention[0].take_glimpses.outputs[0])
            if prefix_attention_in_readout:
                # Name is currently HACKED
                # WORKING: SAME BUG WITH attention output names
                attention_sources.append(self.attention[1].take_glimpses.outputs[0] + '_0')

        else:
            attention_sources = [self.attention.take_glimpses.outputs[0]]

        readout = Readout(
            source_names=['states', 'feedback'] + attention_sources,
            readout_dim=self.vocab_size,
            emitter=SoftmaxEmitter(initial_output=-1, theano_seed=theano_seed),
            feedback_brick=LookupFeedbackWMT15(vocab_size, embedding_dim),
            post_merge=readout_post_merge,
            merged_dim=state_dim)

        # Build sequence generator accordingly
        # TODO: remove the semantic overloading of the `loss_function` kwarg
        # None: intuitively, we want to make sure _not_ to translate things that are already in the target prefix
        print("loss function is: {}".format(loss_function))
        if loss_function == 'cross_entropy':
            # Note: it's the PartialSequenceGenerator which lets us condition upon the target prefix
            self.sequence_generator = PartialSequenceGenerator(
                confidence_model=confidence_model,
                readout=readout,
                transition=self.transition,
                attentions=self.attention,
                fork=Fork([name for name in self.transition.apply.sequences
                           if name != 'mask'], prototype=Linear())
            )
            # the name is important, because it lets us match the brick hierarchy names for the vanilla SequenceGenerator
            # to load pretrained models
            self.sequence_generator.name = 'sequencegenerator'
        elif loss_function == 'min_risk':
            self.sequence_generator = MinRiskPartialSequenceGenerator(
                readout=readout,
                transition=self.transition,
                attentions=self.attention,
                fork=Fork([name for name in self.transition.apply.sequences
                           if name != 'mask'], prototype=Linear())
            )
            # the name is important, because it lets us match the brick hierarchy names for the vanilla SequenceGenerator
            # to load pretrained models
            self.sequence_generator.name = 'sequencegenerator'
        else:
            raise ValueError('The decoder does not support the loss function: {}'.format(loss_function))

        self.children = [self.sequence_generator]

    # @application(inputs=['representation', 'prefix_representation', 'source_sentence_mask',
    #                     'target_sentence_mask', 'target_sentence', 'target_prefix_mask', 'target_prefix'],
    @application(outputs=['cost'])
    def cost(self, rep, source_sentence_mask, prefix_representation,
             target_sentence, target_sentence_mask, target_prefix, target_prefix_mask,
             additional_attention_in_internal_states=True):

        source_sentence_mask = source_sentence_mask.T

        target_sentence = target_sentence.T
        target_sentence_mask = target_sentence_mask.T

        target_prefix = target_prefix.T
        target_prefix_mask = target_prefix_mask.T

        additional_attentions = {}
        if prefix_representation is not None:
            additional_attentions['attended_0'] = prefix_representation
            additional_attentions['attended_mask_0'] = target_prefix_mask

        # Get the cost matrix
        # Note: there is a hard-coded dependency between the 'attended' kwarg and the 'attended' in the recurrent transition
        cost = self.sequence_generator.cost_matrix(**dict_union(
            {
              'mask': target_sentence_mask,
              'outputs': target_sentence,
              'prefix_mask': target_prefix_mask,
              'prefix_outputs': target_prefix,
              'attended': rep,
              'attended_mask': source_sentence_mask,
              'additional_attention_in_internal_states': additional_attention_in_internal_states
            },
            additional_attentions)
        )

        return (cost * target_sentence_mask).sum() / \
               target_sentence_mask.shape[1]


    @application(inputs=['representation', 'source_sentence_mask',
                         'target_sentence_mask', 'target_sentence', 'target_prefix_mask', 'target_prefix'],
                 outputs=['readouts', 'merged_states'])
    def prediction_tags(self, representation, source_sentence_mask,
             target_sentence, target_sentence_mask, target_prefix, target_prefix_mask):

        source_sentence_mask = source_sentence_mask.T
        target_sentence = target_sentence.T
        target_sentence_mask = target_sentence_mask.T
        target_prefix = target_prefix.T
        target_prefix_mask = target_prefix_mask.T

        # Get the cost matrix
        readouts, merged_states = self.sequence_generator.prediction_tags(**{
            'mask': target_sentence_mask,
            'outputs': target_sentence,
            'prefix_mask': target_prefix_mask,
            'prefix_outputs': target_prefix,
            'attended': representation,
            'attended_mask': source_sentence_mask,
        })

        return readouts, merged_states

    @application(inputs=['representation', 'source_sentence_mask',
                         'target_sentence_mask', 'target_sentence', 'target_prefix_mask', 'target_prefix', 'readouts', 'prediction_tags'],
                 outputs=['cost'])
    def confidence_cost(self, representation, source_sentence_mask,
                        target_sentence, target_sentence_mask, target_prefix, target_prefix_mask, readouts, prediction_tags):

        source_sentence_mask = source_sentence_mask.T
        target_sentence = target_sentence.T
        target_sentence_mask = target_sentence_mask.T
        target_prefix = target_prefix.T
        target_prefix_mask = target_prefix_mask.T
        prediction_tags = prediction_tags.T

        # Get the cost matrix
        cost_matrix = self.sequence_generator.confidence_cost(**{
            'mask': target_sentence_mask,
            'outputs': target_sentence,
            'prefix_mask': target_prefix_mask,
            'prefix_outputs': target_prefix,
            'attended': representation,
            'attended_mask': source_sentence_mask,
            'prediction_tags': prediction_tags,
            'merged_states': readouts
        })
        return (cost_matrix * target_sentence_mask).sum() / \
               target_sentence_mask.shape[1]

    # Note: in this formulation, the "target sentence" is actually assumed to be a prediction output by the model
    # Note: a better way would be to output the confidence at each generation step
    @application(inputs=['readouts'],
                 outputs=['confidence_scores'])
    def get_confidence(self, readouts):

        # Get the predictions
        confidence_scores = self.sequence_generator.confidence_predictions(**{
            'readouts': readouts,
        })
        return confidence_scores


    # Note: this requires the decoder to be using sequence_generator which implements expected cost
    # TODO: implement expected cost for target prefix decoding
    @application(inputs=['representation', 'source_sentence_mask',
                         'target_samples_mask', 'target_samples', 'scores'],
                 outputs=['cost'])
    def expected_cost(self, representation, source_sentence_mask, target_samples, target_samples_mask, scores,
                      **kwargs):
        return self.sequence_generator.expected_cost(representation,
                                                     source_sentence_mask,
                                                     target_samples, target_samples_mask, scores, **kwargs)

    # Note the tensor.ones init of the target_prefix_mask is correct in this case
    @application
    def generate(self, source_sentence, representation, **kwargs):
        # if n_steps isn't in kwargs, max prediction len is 2 * max source sentence len in minibatch
        if kwargs.get('n_steps', None) is None:
            kwargs['n_steps'] = 2 * source_sentence.shape[1]

        # NOTE: another dependency between the kwargs for attention -- probably the way to handle this is by passing a list of (attention, mask) pairs
        if kwargs.get('prefix_representation', None) is not None:
            prefix_representation = kwargs['prefix_representation']
            kwargs['attended_0'] = prefix_representation
            target_prefix = kwargs['target_prefix']
            kwargs['attended_mask_0'] = tensor.ones(target_prefix.shape).T
            kwargs['use_additional_attention'] = True

        return self.sequence_generator.generate(
            batch_size=source_sentence.shape[0],
            attended=representation,
            attended_mask=tensor.ones(source_sentence.shape).T,
            **kwargs)



