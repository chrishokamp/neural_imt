import theano
from theano import tensor
from toolz import merge

from blocks.bricks.base import application
from blocks.bricks import NDimensionalSoftmax

from blocks.utils import dict_union, dict_subset
from blocks.bricks.sequence_generators import SequenceGenerator

from blocks.bricks import (Tanh, Maxout, Linear, FeedforwardSequence,
                           Bias, Initializable, MLP)
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.base import application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import GatedRecurrent, Bidirectional
from blocks.bricks.sequence_generators import (
    LookupFeedback, Readout, SoftmaxEmitter,
    SequenceGenerator)
from blocks.roles import add_role, WEIGHT
from blocks.utils import shared_floatx_nans

from machine_translation.model import (InitializableFeedforwardSequence, LookupFeedbackWMT15, GRUInitialState)

# from machine_translation.models import MinRiskSequenceGenerator, PartialSequenceGenerator

from picklable_itertools.extras import equizip

theano.config.optimizer = 'None'
theano.config.traceback.limit = 20

# WORKING: incorporate changes from machine_translation.model here
class Decoder(Initializable):
    """
    Decoder of RNNsearch model.

    Parameters:
    -----------
    vocab_size: int
    embedding_dim: int
    representation_dim: int
    theano_seed: int
    loss_function: str : {'cross_entropy'(default) | 'min_risk'}

    """

    def __init__(self, vocab_size, embedding_dim, state_dim,
                 representation_dim, theano_seed=None, loss_function='cross_entropy', **kwargs):
        super(Decoder, self).__init__(**kwargs)
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
            post_merge=InitializableFeedforwardSequence(
                [Bias(dim=state_dim, name='maxout_bias').apply,
                 Maxout(num_pieces=2, name='maxout').apply,
                 Linear(input_dim=state_dim / 2, output_dim=embedding_dim,
                        use_bias=False, name='softmax0').apply,
                 Linear(input_dim=embedding_dim, name='softmax1').apply]),
            merged_dim=state_dim)

        # Build sequence generator accordingly
        if loss_function == 'cross_entropy':
            self.sequence_generator = SequenceGenerator(
                readout=readout,
                transition=self.transition,
                attention=self.attention,
                fork=Fork([name for name in self.transition.apply.sequences
                           if name != 'mask'], prototype=Linear())
            )
        # TODO: this section is blocking a merge -- working -- move to separate repo
        elif loss_function == 'min_risk':
            # self.sequence_generator = MinRiskSequenceGenerator(
            #     readout=readout,
            #     transition=self.transition,
            #     attention=self.attention,
            #     fork=Fork([name for name in self.transition.apply.sequences
            #                if name != 'mask'], prototype=Linear())
            # )
            self.sequence_generator = PartialSequenceGenerator(
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
                         'target_sentence_mask', 'target_sentence'],
                 outputs=['cost'])
    def cost(self, representation, source_sentence_mask,
             target_sentence, target_sentence_mask):

        source_sentence_mask = source_sentence_mask.T
        target_sentence = target_sentence.T
        target_sentence_mask = target_sentence_mask.T

        # Get the cost matrix
        cost = self.sequence_generator.cost_matrix(**{
            'mask': target_sentence_mask,
            'outputs': target_sentence,
            'attended': representation,
            'attended_mask': source_sentence_mask}
                                                   )

        return (cost * target_sentence_mask).sum() / \
               target_sentence_mask.shape[1]

    # Note: this requires the decoder to be using sequence_generator which implements expected cost
    @application(inputs=['representation', 'source_sentence_mask',
                         'target_samples_mask', 'target_samples', 'scores'],
                 outputs=['cost'])
    def expected_cost(self, representation, source_sentence_mask, target_samples, target_samples_mask, scores,
                      **kwargs):
        return self.sequence_generator.expected_cost(representation,
                                                     source_sentence_mask,
                                                     target_samples, target_samples_mask, scores, **kwargs)


    @application
    def generate(self, source_sentence, representation, **kwargs):
        print(kwargs)
        return self.sequence_generator.generate(
            n_steps=2 * source_sentence.shape[1],
            batch_size=source_sentence.shape[0],
            attended=representation,
            attended_mask=tensor.ones(source_sentence.shape).T,
            **kwargs)


class PartialSequenceGenerator(SequenceGenerator):
    """
    Adds the ability to predict and sample partial target sequences by inputting both a source sequence
    and a target sequence
    """

    def __init__(self, *args, **kwargs):
        super(PartialSequenceGenerator, self).__init__(*args, **kwargs)

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        """
        Use the kwargs to drive the initial states -- if user passed a prefix, then compute all of the states for
        that prefix, then set the initial states for the generator as the final
        states of the prefix
        """

        if 'target_prefix' in kwargs:
            print('target_prefix: {}'.format(kwargs['target_prefix']))
            # Note the transpose
            target_prefix = kwargs['target_prefix'].T
            # TODO: in the batch implementation, each target prefix will have different lengths,
            # TODO: what is the right way to deal with this? -- what are usecases where user would want to
            # TODO: pass prefixes of different lengths, or is this only relevant at training time?
            # TODO: let user pass mask -- get the actual final states using the mask
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

        # TODO: add the target prefix as context??
        # @property
        # def _context_names(self):
        #     existing_contexts = super(PartialSequenceGenerator, self)._context_names
        #     return existing_contexts + ['target_prefix']
