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

# class NamedSequenceContentAttention





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

        self._context_names = list(transition.apply.contexts)
        if add_contexts:
            if not attended_name:
                attended_name = 'attended'
            if not attended_mask_name:
                attended_mask_name = 'attended_mask'
            self._context_names += [attended_name, attended_mask_name]
        # TODO: the else block is currently broken -- doesn't get used
        else:
            attended_name = self._context_names[0]
            attended_mask_name = self._context_names[1]

        if not distribute:
            normal_inputs = [name for name in self._sequence_names
                             if 'mask' not in name]
            # Note: if we change this brick's name or dims, default behavior will break
            distribute = Distribute(normal_inputs,
                                    attentions[0].take_glimpses.outputs[0])

        # add another Distribute for each attention beyond the first one
        additional_attentions = attentions[1:]
        additional_distributes = []
        for i, attention in enumerate(additional_attentions):
            prefix = 'distribute_fork_{}'.format(i)
            # WORKING: HACK TO MATCH OUTPUT NAMES -- SAME BUG AS 'weighted_averages', 'weights' being the same name for all attention bricks
            additional_distribute = Distribute(normal_inputs, attention.take_glimpses.outputs[0] + '_' + str(i), child_prefix=prefix, name='prefix')
            additional_distributes.append(additional_distribute)

        self.transition = transition

        self.attention = attentions[0]
        self.additional_attentions = additional_attentions

        self.distribute = distribute
        self.additional_distributes = additional_distributes

        self.add_contexts = add_contexts

        self.attended_name = attended_name
        self.attended_mask_name = attended_mask_name

        self.preprocessed_attended_name = "preprocessed_" + self.attended_name

        self._glimpse_names = self.attention.take_glimpses.outputs
        # We need to determine which glimpses are fed back.
        # Currently we extract it from `take_glimpses` signature.
        # TODO: is this used anywhere in machine translation?
        self.previous_glimpses_needed = [
            name for name in self._glimpse_names
            if name in self.attention.take_glimpses.inputs]

        # Note:We assume that 'add_contexts' is always true for all additional attentions
        additional_attended_names = []
        additional_preprocessed_attended_names = []
        additional_attended_mask_names = []
        # note that we need to manually set the names of these variables because of the way attention is implemented
        additional_glimpse_names = []
        for i, attention in enumerate(self.additional_attentions):
            additional_attended_name = 'attended_{}'.format(i)
            additional_attended_names.append(additional_attended_name)
            additional_preprocessed_attended_names.append('preprocessed_{}'.format(additional_attended_name))
            additional_attended_mask_name = 'attended_mask_{}'.format(i)
            additional_attended_mask_names.append(additional_attended_mask_name)
            # change the names of the glimpses so that each variable is unique
            additional_glimpse_names.append([n + '_' + str(i) for n in attention.take_glimpses.outputs])
            # add multiple attentions to the contexts
            self._context_names += [additional_attended_name, additional_attended_mask_name]

        self.additional_attended_names = additional_attended_names
        self.additional_preprocessed_attended_names = additional_preprocessed_attended_names
        self.additional_attended_mask_names = additional_attended_mask_names
        self._additional_glimpse_names = additional_glimpse_names
        self.flat_additional_glimpse_names = [n for gs in additional_glimpse_names for n in gs]

        children = [self.transition, self.distribute, self.attention] + self.additional_attentions + self.additional_distributes
        kwargs.setdefault('children', []).extend(children)
        super(MultipleAttentionRecurrent, self).__init__(**kwargs)

    def _push_allocation_config(self):
        self.attention.state_dims = self.transition.get_dims(self.attention.state_names)
        self.attention.attended_dim = self.get_dim(self.attended_name)
        for attention, attended_name in zip(self.additional_attentions, self.additional_attended_names):
            attention.state_dims = self.transition.get_dims(attention.state_names)
            attention.attended_dim = self.get_dim(attended_name)

        self.distribute.source_dim = self.attention.get_dim(
            self.distribute.source_name)
        self.distribute.target_dims = self.transition.get_dims(
            self.distribute.target_names)
        for distribute, attention in zip(self.additional_distributes, self.additional_attentions):
            # WORKING HACK HACK HACK -- attention brick cannot currently get renamed dim
            # distribute.source_dim = attention.get_dim(distribute.source_name)
            distribute.source_dim = attention.get_dim(distribute.source_name[:-2])
            distribute.target_dims = self.transition.get_dims(distribute.target_names)

    @application
    def take_glimpses(self, **kwargs):
        r"""Compute glimpses with the base attention and all additional attention mechanisms.

        A thin wrapper over each `self.attention.take_glimpses`: takes care
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
        # Note: previous glimpses are currently never needed in machine translation
        glimpses_needed = dict_subset(glimpses, self.previous_glimpses_needed)

        # result for the default attention
        # WORKING: why is 'preprocessed_attended_name' sometimes None in the kwargs?
        # WORKING: -- where is take_glimses getting called from?
        default_attended =  kwargs.pop(self.attended_name)
        result = self.attention.take_glimpses(
            default_attended,
            kwargs.pop(self.preprocessed_attended_name, None),
            kwargs.pop(self.attended_mask_name, None),
            **dict_union(states, glimpses_needed))

        additional_results = []
        # if len(set(kwargs.keys()).intersection(set(self.additional_attended_names))) > 0:
        if kwargs.get('use_additional_attention', False):
            for tup in zip(self.additional_attentions,
                           self.additional_attended_names,
                           self.additional_preprocessed_attended_names,
                           self.additional_attended_mask_names):

                attention, attended_name, preprocessed_attended_name, mask_name = tup
                additional_result = attention.take_glimpses(kwargs.pop(attended_name),
                                                            kwargs.pop(preprocessed_attended_name, None),
                                                            kwargs.pop(mask_name, None),
                                                            **dict_union(states, glimpses_needed))
                additional_results.extend(additional_result)
        else:
            # just add dummy values for everything
            for _ in self.additional_attentions:
                additional_results.extend([tensor.zeros(result[0].shape), tensor.zeros(result[1].shape)])


        # At this point kwargs may contain additional items.
        # e.g. AttentionRecurrent.transition.apply.contexts
        # for each attention, we have [attention_take_glimpses_weighted_averages, attention_take_glimpses_weights]
        return result + additional_results

    @take_glimpses.property('outputs')
    def take_glimpses_outputs(self):
        return self._glimpse_names + self.flat_additional_glimpse_names

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
        # TODO: how did these kwargs get here in the first place, what is the graph??
        # TODO: will we need initial_glimpses for every attention? -- probably yes
        if 'initial_state_context' in kwargs:
            kwargs.pop('initial_state_context')
        if 'initial_glimpses' in kwargs:
            kwargs.pop('initial_glimpses')
        if 'initial_states' in kwargs:
            kwargs.pop('initial_states')


        # if user provided additional attentions, update with multiple glimpses from multiple attentions
        # update sequences for each attention (sum distribute brick output with current sequence representation
        # WORKING: how to decide whether or not to apply additional attentions?
        use_additional_attention = False
        if len(set(kwargs.keys()).intersection(set(self.additional_attended_names))) > 0:
            use_additional_attention = True
            for distribute, additional_glimpses, attended_name, attended_mask_name in zip(self.additional_distributes,
                                                                                          self._additional_glimpse_names,
                                                                                          self.additional_attended_names,
                                                                                          self.additional_attended_mask_names):
                current_glimpses = dict_subset(kwargs, additional_glimpses, pop=True)
                # Note that we skip the "add_contexts" logic above, and just assume it's true
                kwargs.pop(attended_name)
                kwargs.pop(attended_mask_name, None)
                # apply the current attention
                sequences.update(distribute.apply(
                                 as_dict=True, **dict_subset(dict_union(sequences, current_glimpses),
                                 distribute.apply.inputs)))


        # Finally apply the default attention
        sequences.update(self.distribute.apply(
            as_dict=True, **dict_subset(dict_union(sequences, glimpses),
                                        self.distribute.apply.inputs)))

        # WORKING we must pop if necessary -- if the glimpses from the initial state didn't get used
        if not use_additional_attention:
            for glimpse_names in self._additional_glimpse_names:
                for g in glimpse_names:
                    kwargs.pop(g)

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

        # WORKING: the use of additional attentions must be optional for each apply call
        additional_attentions = {}
        additional_glimpses = {}
        use_additional_attention = False
        if len(set(kwargs.keys()).intersection(set(self.additional_attended_names))) > 0:
            additional_attended = {name: kwargs[name] for name in self.additional_attended_names}
            preprocessed_additional_attended = {name: kwargs.pop(name) for name in self.additional_preprocessed_attended_names}
            additional_attended_mask = {name: kwargs.get(name) for name in self.additional_attended_mask_names}
	    # put all of the dicts together
            additional_attentions = dict_union(additional_attended, preprocessed_additional_attended, additional_attended_mask)
            use_additional_attention = True

        # Note: weighted_averages and weights need to have different names for each attention block
        # WORKING: we always pop here, because they're in the initial states
        try:
            additional_glimpses = dict_union(*[dict_subset(kwargs, glimpse_names, pop=True)
                                             for glimpse_names in self._additional_glimpse_names])
        except:
            import ipdb; ipdb.set_trace()

        # WORKING: make sure multiple glimpses from take_glimpses are handled at prediction time as well
        # WORKING HERE
        current_glimpses = self.take_glimpses(
            as_dict=True,
            use_additional_attention=use_additional_attention,
            **dict_union(
                states, glimpses, additional_glimpses,
                {self.attended_name: attended,
                 self.attended_mask_name: attended_mask,
                 self.preprocessed_attended_name: preprocessed_attended},
                additional_attentions
            )
        )
        current_states = self.compute_states(
            as_list=True,
            **dict_union(sequences, states, current_glimpses, kwargs))
        return current_states + list(current_glimpses.values())

    @do_apply.property('sequences')
    def do_apply_sequences(self):
        return self._sequence_names

    @do_apply.property('contexts')
    def do_apply_contexts(self):
        return self._context_names + [self.preprocessed_attended_name] + self.additional_preprocessed_attended_names

    # NOTE that we don't actually use the glimpses as states in current machine translation
    @do_apply.property('states')
    def do_apply_states(self):
        return self._state_names + self._glimpse_names + self.flat_additional_glimpse_names

    @do_apply.property('outputs')
    def do_apply_outputs(self):
        return self._state_names + self._glimpse_names + self.flat_additional_glimpse_names

    @application
    def apply(self, **kwargs):
        """Preprocess a sequence attending the attended context at every step.

        Preprocesses the attended context and runs :meth:`do_apply`. See
        :meth:`do_apply` documentation for further information.

        """
        preprocessed_attended = self.attention.preprocess(kwargs[self.attended_name])
        additional_preprocessed_attended = {}

        if len(set(kwargs.keys()).intersection(set(self.additional_attended_names))) > 0:
            for attention, attended_name, preprocessed_attended_name in zip(self.additional_attentions,
                                                                            self.additional_attended_names,
                                                                            self.additional_preprocessed_attended_names):
                additional_preprocessed_attended[preprocessed_attended_name] = attention.preprocess(kwargs[attended_name])

        return self.do_apply(**dict_union(kwargs,
                                          {self.preprocessed_attended_name: preprocessed_attended},
                                          additional_preprocessed_attended)
                             )

    @apply.delegate
    def apply_delegate(self):
        # TODO: Nice interface for this trick?
        return self.do_apply.__get__(self, None)

    @apply.property('contexts')
    def apply_contexts(self):
        return self._context_names

    # Note: this method is currently overridden in initial state attention recurrent
    @application
    def initial_states(self, batch_size, **kwargs):

        initial_glimpses = self.attention.initial_glimpses(batch_size, kwargs[self.attended_name])

        additional_initial_glimpses = []

        # Note: we always return all initial states, just don't use them in transitions that don't require additional attention
        # Note: this can cause bugs when these states are used to initialize the real transition!
        for attention, attended_name in zip(self.additional_attentions, self.additional_attended_names):

            if attended_name in kwargs:
                initial_glimpses = attention.initial_glimpses(batch_size, kwargs[attended_name])
                additional_initial_glimpses.extend(pack(initial_glimpses))
            else:
                # Note: the dimensions still need to be the same as the dummy outputs from `take_glimpses`
                additional_initial_glimpses.extend(self.attention.initial_glimpses(batch_size, kwargs[self.attended_name]))

        initial_states = (pack(self.transition.initial_states(batch_size, **kwargs)) +
                          pack(initial_glimpses) +
                          pack(additional_initial_glimpses))

        return initial_states

    # WORKING: this method is currently overridden in initial state attention recurrent
    @initial_states.property('outputs')
    def initial_states_outputs(self):
        return self.do_apply.states

    def get_dim(self, name):
        # WORKING: do 'weighted_averages' and 'weights' get duplicated for every attention brick? -- yes, this breaks
        # if name in self.flat_additional_glimpse_names:
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
            if name in self.additional_attended_names:
                attention = self.additional_attentions[self.additional_attended_names.index(name)]
                # this is a hack so that AbstractAttention can find the dimension
                if name.endswith('_0'):
                    temp_name = name[:-2]
                return attention.get_dim(temp_name)
            if name in self.flat_additional_glimpse_names:
                attention_idx = self.flat_additional_glimpse_names.index(name)
                if attention_idx % 2 == 1:
                    attention_idx -= 1
                attention = self.additional_attentions[attention_idx]
                if 'weighted_averages' in name:
                    return attention.attended_dim
                if 'weights' in name:
                    return 0
            if name in self.additional_attended_mask_names:
                return 0
        return self.transition.get_dim(name)
