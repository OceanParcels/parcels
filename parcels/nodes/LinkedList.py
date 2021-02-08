from parcels.nodes.Node import *
from sortedcontainers import SortedList
from numpy import int32, int64, uint32, uint64
# from copy import copy
from copy import deepcopy
# import gc


# ========================== #
# = Verdict: nice try, but = #
# = overrides to the del() = #
# = function won't work.   = #
# ========================== #
class RealList(SortedList):
    dtype = None

    def __init__(self, iterable=None, dtype=Node):
        super(RealList, self).__init__(iterable)
        self.dtype = dtype

    def __del__(self):
        self.clear()

    def clear(self):
        """Remove all the elements from the list."""
        n = self.__len__()
        # print("# remaining items: {}".format(n))
        if n > 0:
            # print("Deleting {} elements ...".format(n))
            while (n > 0):
                # val = self.pop(); del val
                # super().__delitem__(n-1)
                # self.pop()

                self.__getitem__(-1).unlink()
                self.__delitem__(-1)
                n = self.__len__()
                # print("Deleting {} elements ...".format(n))
        # gc.collect()
        super()._clear()

    def __new__(cls, iterable=None, key=None, load=1000, dtype=Node):
        return object.__new__(cls)

    def add(self, val):
        assert type(val) == self.dtype
        if isinstance(val, Node):
            self._add_by_node(val)
        elif isinstance(val, int) or type(val) in [int32, uint32, int64, uint64]:
            self._add_by_id(val)
        else:
            self._add_by_pdata(val)

    def _add_by_node(self, val):
        n = self.__len__()
        index = self.bisect_right(val)
        if index < n:
            next_node = self.__getitem__(index)
        else:
            next_node = None
        if index > 0:
            prev_node = self.__getitem__(index - 1)
        else:
            prev_node = None
        if next_node is not None:
            next_node.set_prev(val)
            val.set_next(next_node)
        if prev_node is not None:
            prev_node.set_next(val)
            val.set_prev(prev_node)
        super().add(val)

    def _add_by_id(self, val):
        n = self.__len__()
        index = self.bisect_right(val)
        if index < n:
            next_node = self.__getitem__(index)
        else:
            next_node = None
        if index > 0:
            prev_node = self.__getitem__(index - 1)
        else:
            prev_node = None
        new_node = self.dtype(prev=prev_node, next=next_node, id=val)
        super().add(new_node)

    def _add_by_pdata(self, val):
        new_node = self.dtype(data=val)
        super().add(new_node)
        n = self.__len__()
        index = self.index(new_node)
        if index < (n - 2):
            next_node = self.__getitem__(index + 1)
        else:
            next_node = None
        if index > 0:
            prev_node = self.__getitem__(index - 1)
        else:
            prev_node = None
        if next_node is not None:
            next_node.set_prev(new_node)
            new_node.set_next(next_node)
        if prev_node is not None:
            prev_node.set_next(new_node)
            new_node.set_prev(prev_node)

    def append(self, val):
        self.add(val)

    def pop(self, idx=-1, deepcopy_elem=False):
        """
        Because we expect the return node to be of use,
        the actual node is NOT physically deleted (by calling
        the destructor). pop() only dereferences the object
        in the list. The parameter 'deepcopy_elem' can be set so as
        to physically delete the list object and return a deep copy (unlisted).
        :param idx:
        :param deepcopy_elem:
        :return:
        """
        if deepcopy_elem:
            val = super().pop(idx)
            result = deepcopy(val)
            # result = deepcopy(self.__getitem__(idx))
            del val
            return result
        return super().pop(idx)
