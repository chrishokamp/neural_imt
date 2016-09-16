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

# from machine_translation.models import MinRiskSequenceGenerator, PartialSequenceGenerator

from picklable_itertools.extras import equizip

# theano.config.optimizer = 'None'
# theano.config.traceback.limit = 20

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

    def __init__(self, readout, transition, attention,
                 add_contexts=True, confidence_model=None, **kwargs):
        normal_inputs = [name for name in transition.apply.sequences
                         if 'mask' not in name]
        kwargs.setdefault('fork', Fork(normal_inputs))
        transition = InitialStateAttentionRecurrent(
            transition, attention,
            add_contexts=add_contexts, name="att_trans")

        self.softmax = NDimensionalSoftmax()
        super(PartialSequenceGenerator, self).__init__(
            readout, transition, **kwargs)
        self.children.append(self.softmax)

        # Working: add the option to include a next-word confidence model
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

        next_glimpses = self.transition.take_glimpses(
            as_dict=True, **dict_union(states, glimpses, contexts))
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
            **dict_union(next_inputs, states, next_glimpses, contexts))

        # TODO: switch to directly getting the probs from softmax
        next_probs = self.softmax.apply(next_readouts)

        # working: optionally also query the confidence model
        # next_merged_states = self.readout.merged_states(
        #     feedback=self.readout.feedback(outputs),
        #     **dict_union(states, next_glimpses, contexts))
        # next_confidences = self.confidence_model.apply(next_merged_states)

        # return (next_states + [next_outputs] +
        #         list(next_glimpses.values()) + [next_costs] + [next_probs] + [next_confidences])
        return (next_states + [next_outputs] +
                list(next_glimpses.values()) + [next_costs] + [next_probs])

    @generate.delegate
    def generate_delegate(self):
        return self.transition.apply

    @generate.property('states')
    def generate_states(self):
        return self._state_names + ['outputs'] + self._glimpse_names

    @generate.property('outputs')
    def generate_outputs(self):
        return (self._state_names + ['outputs'] +
                self._glimpse_names + ['costs'] + ['word_probs'])

    def get_dim(self, name):
        if name in (self._state_names + self._context_names +
                        self._glimpse_names):
            return self.transition.get_dim(name)
        elif name == 'outputs':
            return self.readout.get_dim(name)
        return super(BaseSequenceGenerator, self).get_dim(name)

    # Note: this function is only used by self.generate, because it has the recurrent decorator
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
            # TODO: mask should be optional so that this method can be used transparently for prediction only
            # TODO: how is the attended mask handled?
            mask = None

            # Prepare input for the iterative part
            states = dict_subset(kwargs, self._state_names, must_have=False)

            # masks in context are optional (e.g. `attended_mask`)
            contexts = dict_subset(kwargs, self._context_names, must_have=False)

            feedback = self.readout.feedback(target_prefix)
            inputs = self.fork.apply(feedback, as_dict=True)

            # Run the recurrent network
            results = self.transition.apply(
                mask=mask, return_initial_states=True, as_dict=True,
                **dict_union(inputs, states, contexts))

            # Remember, glimpses are computed _before_ output stage, states are
            # computed after.
            states = {name: results[name] for name in self._state_names}
            # this is just to keep the 'batch_size' parameter in the graph, because it's used by beam search in blocks
            states['states'] = states['states'].reshape((states['states'].shape[0],
                                                         batch_size, states['states'].shape[2]))

            glimpses = {name: results[name] for name in self._glimpse_names}

            # the initial states of the sequence generator are:
            # ['states', 'outputs', 'weighted_averages', 'weights'] (the last two are in the glimpses)
            state_dict = {
                'states': states['states'][-1],
                'outputs':  target_prefix[-1],
                'weighted_averages': glimpses['weighted_averages'][-1],
                'weights': glimpses['weights'][-1]
            }

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
            # feedback[0],
            # self.readout.feedback(self.readout.initial_outputs(batch_size)))
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

    # WORKING HERE -- get confidence model outputs and compare to the references from the datastream, compute binary CE as cost
    @application
    def confidence_cost(self, application_call, outputs, prefix_outputs, prediction_tags, merged_states, mask=None, prefix_mask=None, **kwargs):

        #merged_states = self.get_merged_states(outputs, prefix_outputs, mask, prefix_mask, **kwargs)
        #time_major_shape = merged_states.shape
        #merged_states = merged_states.reshape((time_major_shape[0]*time_major_shape[1], time_major_shape[2]))
        confidence = theano.tensor.nnet.sigmoid(self.confidence_model.apply(merged_states))
        # readouts  = prediction_tags.dimshuffle(2,1,0)
        # flatten

        # confidence = theano.tensor.nnet.sigmoid(self.confidence_model.apply(readouts))

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
        merged_states = self.get_merged_states(outputs, prefix_outputs, mask, prefix_mask, **kwargs)

        # WORKING: there is a problem with the gradient computation from this function
        # WORKING: split cost computation out of prediction
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

        #return readouts,y_equal
        #return y_equal
        #return y_emissions
        #return confidence_cost

        return readouts, merged_states
        #return y_true




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

class InitialStateAttentionRecurrent(AttentionRecurrent):
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
        if 'initial_states' in kwargs and 'initial_glimpses' in kwargs:
            # Note: these states currently get popped out of AttentionRecurrent kwargs in the same way as mmmt

            transition_initial_states = kwargs.pop('initial_states')
            attention_initial_glimpses = kwargs.pop('initial_glimpses')
            initial_states = (pack(transition_initial_states) + pack(attention_initial_glimpses))
        else:
            initial_states = (pack(self.transition.initial_states(batch_size, **kwargs)) +
                              pack(self.attention.initial_glimpses(batch_size, kwargs[self.attended_name])))

        return initial_states

    @initial_states.property('outputs')
    def initial_states_outputs(self):
        return self.do_apply.states


# WORKING: make the prefix representation configurable, so that we can swap out modules (forward recurrent, bidir, attention)
# WORKING: make sure the prefix mask is handled correctly
# TODO: change sequence generator transition to InitialStateAttentionRecurrent
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
                 representation_dim, theano_seed=None, loss_function='cross_entropy', use_post_merge=True,
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

        # Initialize the attention mechanism
        self.attention = SequenceContentAttention(
            state_names=self.transition.apply.states,
            attended_dim=representation_dim,
            match_dim=state_dim, name="attention")


        # WORKING: add the confidence model
        confidence_model = InitializableFeedforwardSequence([
                 # Linear(input_dim=vocab_size, output_dim=1000,
                 #        use_bias=True, name='confidence_model0').apply,
                 Bias(state_dim).apply,
                 Tanh(name='confidence_tanh').apply,
                 Linear(input_dim=state_dim, output_dim=1, use_bias=True, name='confidence_model1').apply])
        # END WORKING: add the confidence model


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

        import ipdb; ipdb.set_trace()

        # Initialize the readout, note that SoftmaxEmitter emits -1 for
        # initial outputs which is used by LookupFeedBackWMT15
        readout = Readout(
            source_names=['states', 'feedback',
                          # Chris: it's key that we're taking the first output of self.attention.take_glimpses.outputs
                          # Chris: the first output is the weighted avgs, the second is the weights in (batch, time)
                          self.attention.take_glimpses.outputs[0]],
            readout_dim=self.vocab_size,
            emitter=SoftmaxEmitter(initial_output=-1, theano_seed=theano_seed),
            feedback_brick=LookupFeedbackWMT15(vocab_size, embedding_dim),
            post_merge=readout_post_merge,
            merged_dim=state_dim)

        # Build sequence generator accordingly
        # TODO: remove the semantic overloading of the `loss_function` kwarg
        # TODO: BIG TIME HACK HERE
        # WORKING: add confidence model for IMT
        if loss_function == 'cross_entropy':
            # Note: it's the PartialSequenceGenerator which lets us condition upon the target prefix
            self.sequence_generator = PartialSequenceGenerator(
                confidence_model=confidence_model,
                readout=readout,
                transition=self.transition,
                attention=self.attention,
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
                attention=self.attention,
                fork=Fork([name for name in self.transition.apply.sequences
                           if name != 'mask'], prototype=Linear())
            )
            # the name is important, because it lets us match the brick hierarchy names for the vanilla SequenceGenerator
            # to load pretrained models
            self.sequence_generator.name = 'sequencegenerator'
        else:
            raise ValueError('The decoder does not support the loss function: {}'.format(loss_function))

        self.children = [self.sequence_generator]

    @application(inputs=['representation', 'source_sentence_mask',
                         'target_sentence_mask', 'target_sentence', 'target_prefix_mask', 'target_prefix'],
                 outputs=['cost'])
    def cost(self, representation, source_sentence_mask,
             target_sentence, target_sentence_mask, target_prefix, target_prefix_mask):

        source_sentence_mask = source_sentence_mask.T
        target_sentence = target_sentence.T
        target_sentence_mask = target_sentence_mask.T
        target_prefix = target_prefix.T
        target_prefix_mask = target_prefix_mask.T

        # Get the cost matrix
        cost = self.sequence_generator.cost_matrix(**{
            'mask': target_sentence_mask,
            'outputs': target_sentence,
            'prefix_mask': target_prefix_mask,
            'prefix_outputs': target_prefix,
            'attended': representation,
            'attended_mask': source_sentence_mask,
        }
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

        # transpose because tags are (time, batch)
        #return tags.T
        # WORKING: actually return the softmax output, not the tags
        return readouts, merged_states

    # WORKING: implement word-level confidence cost
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

    # WORKING: implement word-level confidence cost
    # WORKING: in this formulation, the "target sentence" is actually assumed to be a prediction output by the model
    # WORKING: a better way would be to output the confidence at each generation step
    @application(inputs=['representation', 'source_sentence_mask',
                         'target_sentence_mask', 'target_sentence', 'target_prefix_mask', 'target_prefix'],
                 outputs=['confidence_scores'])
    def get_confidence(self, representation, source_sentence_mask,
                        target_sentence, target_sentence_mask, target_prefix, target_prefix_mask):

        source_sentence_mask = source_sentence_mask.T
        target_sentence = target_sentence.T
        target_sentence_mask = target_sentence_mask.T
        target_prefix = target_prefix.T
        target_prefix_mask = target_prefix_mask.T

        # Get the predictions
        confidence_scores = self.sequence_generator.confidence_predictions(**{
            'mask': target_sentence_mask,
            'outputs': target_sentence,
            'prefix_mask': target_prefix_mask,
            'prefix_outputs': target_prefix,
            'attended': representation,
            'attended_mask': source_sentence_mask,
        })
        return confidence_scores


    # Note: this requires the decoder to be using sequence_generator which implements expected cost
    # WORKING: implement expected cost for target prefix decoding
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
        return self.sequence_generator.generate(
            n_steps=2 * source_sentence.shape[1],
            batch_size=source_sentence.shape[0],
            attended=representation,
            attended_mask=tensor.ones(source_sentence.shape).T,
            **kwargs)



