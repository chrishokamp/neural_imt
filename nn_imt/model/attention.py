from theano import tensor

from blocks.bricks import Initializable
from blocks.bricks.base import application
from blocks.bricks.parallel import Distribute
from blocks.bricks.recurrent import recurrent
from blocks.utils import dict_union, dict_subset, pack

from blocks.bricks.attention import AbstractAttentionRecurrent

# WORKING: subclass AttentionRecurrent?
# WORKING -- code buffer

# @lazy(allocation=['input_names', 'input_dims', 'output_dims'])
# def __init__(self, input_names, input_dims, output_dims,
#              prototype, child_prefix=None, **kwargs):
#     super(Parallel, self).__init__(**kwargs)
#     if not child_prefix:
#         child_prefix = "transform"
#
#     self.input_names = input_names
#     self.input_dims = input_dims
#     self.output_dims = output_dims
#     self.prototype = prototype
#
#     self.children = []
#     for name in input_names:
#         self.children.append(copy.deepcopy(self.prototype))
#         self.children[-1].name = "{}_{}".format(child_prefix, name)
#
# def _push_allocation_config(self):
#     for input_dim, output_dim, child in \
#             equizip(self.input_dims, self.output_dims, self.children):
#         child.input_dim = input_dim
#         child.output_dim = output_dim



class MultipleAttentionRecurrent(AbstractAttentionRecurrent, Initializable):
    """Combines an attention mechanism and a recurrent transition.

    This brick equips a recurrent transition with an attention mechanism.
    In order to do this two more contexts are added: one to be attended and
    a mask for it. It is also possible to use the contexts of the given
    recurrent transition for these purposes and not add any new ones,
    see `add_context` parameter.

    At the beginning of each step attention mechanism produces glimpses;
    these glimpses together with the current states are used to compute the
    next state and finish the transition. In some cases glimpses from the
    previous steps are also necessary for the attention mechanism, e.g.
    in order to focus on an area close to the one from the previous step.
    This is also supported: such glimpses become states of the new
    transition.

    To let the user control the way glimpses are used, this brick also
    takes a "distribute" brick as parameter that distributes the
    information from glimpses across the sequential inputs of the wrapped
    recurrent transition.

    Parameters
    ----------
    transition : :class:`.BaseRecurrent`
        The recurrent transition.
    attention : :class:`~.bricks.Brick`
        The attention mechanism.
    distribute : :class:`~.bricks.Brick`, optional
        Distributes the information from glimpses across the input
        sequences of the transition. By default a :class:`.Distribute` is
        used, and those inputs containing the "mask" substring in their
        name are not affected.
    add_contexts : bool, optional
        If ``True``, new contexts for the attended and the attended mask
        are added to this transition, otherwise existing contexts of the
        wrapped transition are used. ``True`` by default.
    attended_name : str
        The name of the attended context. If ``None``, "attended"
        or the first context of the recurrent transition is used
        depending on the value of `add_contents` flag.
    attended_mask_name : str
        The name of the mask for the attended context. If ``None``,
        "attended_mask" or the second context of the recurrent transition
        is used depending on the value of `add_contents` flag.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    Wrapping your recurrent brick with this class makes all the
    states mandatory. If you feel this is a limitation for you, try
    to make it better! This restriction does not apply to sequences
    and contexts: those keep being as optional as they were for
    your brick.

    Those coming to Blocks from Groundhog might recognize that this is
    a `RecurrentLayerWithSearch`, but on steroids :)

    """
    def __init__(self, transition, attentions, distribute=None,
                 add_contexts=True,
                 attended_name=None, attended_mask_name=None,
                 **kwargs):
        if type(attentions) is not list:
            raise TypeError("MultipleAttentionRecurrent needs a list of contexts to compute attention over")

        self._sequence_names = list(transition.apply.sequences)
        self._state_names = list(transition.apply.states)

        # WORKING: add multiple attentions to the contexts
        # WORKING: the transition gets the contexts via self._context_names(??)
        self._context_names = list(transition.apply.contexts)
        if add_contexts:
            if not attended_name:
                attended_name = 'attended'
            if not attended_mask_name:
                attended_mask_name = 'attended_mask'
            self._context_names += [attended_name, attended_mask_name]
        else:
            attended_name = self._context_names[0]
            attended_mask_name = self._context_names[1]
        # WORKING: how to distribute the output of multiple attentions??
        # WORKING: a simple option would be to concat the attentions, then apply the distribute
        if not distribute:
            normal_inputs = [name for name in self._sequence_names
                             if 'mask' not in name]
            # WORKING: keep legacy behavior, warn user when things look weird
            distribute = Distribute(normal_inputs,
                                    attentions[0].take_glimpses.outputs[0])

        self.transition = transition

        # WORKING: keep legacy behavior, warn user when things look weird
        self.attention = attentions[0]
        self.attentions = attentions

        self.distribute = distribute
        self.add_contexts = add_contexts
        self.attended_name = attended_name
        self.attended_mask_name = attended_mask_name

        import ipdb; ipdb.set_trace()

        self.preprocessed_attended_name = "preprocessed_" + self.attended_name

        self._glimpse_names = self.attention.take_glimpses.outputs
        # We need to determine which glimpses are fed back.
        # Currently we extract it from `take_glimpses` signature.
        self.previous_glimpses_needed = [
            name for name in self._glimpse_names
            if name in self.attention.take_glimpses.inputs]

        children = [self.transition, self.distribute] + self.attentions
        kwargs.setdefault('children', []).extend(children)
        super(MultipleAttentionRecurrent, self).__init__(**kwargs)

    def _push_allocation_config(self):
        for attention in self.attentions:
            attention.state_dims = self.transition.get_dims(attention.state_names)
            attention.attended_dim = self.get_dim(self.attended_name)

        # WORKING: distribute still only accepts a single attention, concat the output of all attentions together
        self.distribute.source_dim = self.attention.get_dim(
            self.distribute.source_name)
        self.distribute.target_dims = self.transition.get_dims(
            self.distribute.target_names)

    @application
    def take_glimpses(self, **kwargs):
        r"""Compute glimpses with the attention mechanism.

        A thin wrapper over `self.attention.take_glimpses`: takes care
        of choosing and renaming the necessary arguments.

        Parameters
        ----------
        \*\*kwargs
            Must contain the attended, previous step states and glimpses.
            Can optionaly contain the attended mask and the preprocessed
            attended.

        Returns
        -------
        glimpses : list of :class:`~tensor.TensorVariable`
            Current step glimpses.

        """
        states = dict_subset(kwargs, self._state_names, pop=True)
        glimpses = dict_subset(kwargs, self._glimpse_names, pop=True)
        glimpses_needed = dict_subset(glimpses, self.previous_glimpses_needed)
        # WORKING: for brick, attended_name in zip(attention_bricks, attended_contexts)
        # WORKING: rename to 'results'
        result = self.attention.take_glimpses(
            kwargs.pop(self.attended_name),
            kwargs.pop(self.preprocessed_attended_name, None),
            kwargs.pop(self.attended_mask_name, None),
            **dict_union(states, glimpses_needed))
        # At this point kwargs may contain additional items.
        # e.g. AttentionRecurrent.transition.apply.contexts
        return result

    @take_glimpses.property('outputs')
    def take_glimpses_outputs(self):
        return self._glimpse_names

    @application
    def compute_states(self, **kwargs):
        r"""Compute current states when glimpses have already been computed.

        Combines an application of the `distribute` that alter the
        sequential inputs of the wrapped transition and an application of
        the wrapped transition. All unknown keyword arguments go to
        the wrapped transition.

        Parameters
        ----------
        \*\*kwargs
            Should contain everything what `self.transition` needs
            and in addition the current glimpses.

        Returns
        -------
        current_states : list of :class:`~tensor.TensorVariable`
            Current states computed by `self.transition`.

        """
        # make sure we are not popping the mask
        normal_inputs = [name for name in self._sequence_names
                         if 'mask' not in name]
        sequences = dict_subset(kwargs, normal_inputs, pop=True)
        glimpses = dict_subset(kwargs, self._glimpse_names, pop=True)
        if self.add_contexts:
            kwargs.pop(self.attended_name)
            # attended_mask_name can be optional
            kwargs.pop(self.attended_mask_name, None)

        # Chris: hack to avoid kwarg error
        # WORKING: how did these kwargs get here in the first place, what is the graph??
        if 'initial_state_context' in kwargs:
            kwargs.pop('initial_state_context')
        if 'initial_glimpses' in kwargs:
            kwargs.pop('initial_glimpses')
        if 'initial_states' in kwargs:
            kwargs.pop('initial_states')

        # WORKING: update with multiple glimpses from multiple attentions
        sequences.update(self.distribute.apply(
            as_dict=True, **dict_subset(dict_union(sequences, glimpses),
                                        self.distribute.apply.inputs)))
        current_states = self.transition.apply(
            iterate=False, as_list=True,
            **dict_union(sequences, kwargs))
        return current_states

    @compute_states.property('outputs')
    def compute_states_outputs(self):
        return self._state_names

    @recurrent
    def do_apply(self, **kwargs):
        r"""Process a sequence attending the attended context every step.

        In addition to the original sequence this method also requires
        its preprocessed version, the one computed by the `preprocess`
        method of the attention mechanism. Unknown keyword arguments
        are passed to the wrapped transition.

        Parameters
        ----------
        \*\*kwargs
            Should contain current inputs, previous step states, contexts,
            the preprocessed attended context, previous step glimpses.

        Returns
        -------
        outputs : list of :class:`~tensor.TensorVariable`
            The current step states and glimpses.

        """
        attended = kwargs[self.attended_name]
        preprocessed_attended = kwargs.pop(self.preprocessed_attended_name)

        attended_mask = kwargs.get(self.attended_mask_name)
        sequences = dict_subset(kwargs, self._sequence_names, pop=True,
                                must_have=False)
        states = dict_subset(kwargs, self._state_names, pop=True)
        glimpses = dict_subset(kwargs, self._glimpse_names, pop=True)

        # WORKING: take_glimpses now returns multiple glimpses
        # WORKING: make sure this is handled at prediction time as well
        # WORKING: make sure glimpses are added to the contexts
        current_glimpses = self.take_glimpses(
            as_dict=True,
            **dict_union(
                states, glimpses,
                {self.attended_name: attended,
                 self.attended_mask_name: attended_mask,
                 self.preprocessed_attended_name: preprocessed_attended}))
        current_states = self.compute_states(
            as_list=True,
            **dict_union(sequences, states, current_glimpses, kwargs))
        return current_states + list(current_glimpses.values())

    @do_apply.property('sequences')
    def do_apply_sequences(self):
        return self._sequence_names

    @do_apply.property('contexts')
    def do_apply_contexts(self):
        return self._context_names + [self.preprocessed_attended_name]

    @do_apply.property('states')
    def do_apply_states(self):
        return self._state_names + self._glimpse_names

    @do_apply.property('outputs')
    def do_apply_outputs(self):
        return self._state_names + self._glimpse_names

    @application
    def apply(self, **kwargs):
        """Preprocess a sequence attending the attended context at every step.

        Preprocesses the attended context and runs :meth:`do_apply`. See
        :meth:`do_apply` documentation for further information.

        """
        preprocessed_attended = self.attention.preprocess(
            kwargs[self.attended_name])
        return self.do_apply(
            **dict_union(kwargs,
                         {self.preprocessed_attended_name:
                              preprocessed_attended}))

    @apply.delegate
    def apply_delegate(self):
        # TODO: Nice interface for this trick?
        return self.do_apply.__get__(self, None)

    @apply.property('contexts')
    def apply_contexts(self):
        return self._context_names

    @application
    def initial_states(self, batch_size, **kwargs):
        return (pack(self.transition.initial_states(
            batch_size, **kwargs)) +
                pack(self.attention.initial_glimpses(
                    batch_size, kwargs[self.attended_name])))

    @initial_states.property('outputs')
    def initial_states_outputs(self):
        return self.do_apply.states

    def get_dim(self, name):
        # WORKING: there are multiple attentions, we need to find the right one for this name
        if name in self._glimpse_names:
            return self.attention.get_dim(name)
        if name == self.preprocessed_attended_name:
            (original_name,) = self.attention.preprocess.outputs
            return self.attention.get_dim(original_name)
        if self.add_contexts:
            if name == self.attended_name:
                return self.attention.get_dim(
                    self.attention.take_glimpses.inputs[0])
            if name == self.attended_mask_name:
                return 0
        return self.transition.get_dim(name)
