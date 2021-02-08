import ctypes
import sys
import os
from parcels.tools import idgen
from parcels.wrapping import *
from numpy import int32, int64, uint32, uint64
import random


class Node(object):
    prev = None
    next = None
    id = None
    data = None
    registered = False

    def __init__(self, prev=None, next=None, id=None, data=None):
        self.registered = True
        if prev is not None:
            assert (isinstance(prev, Node))
            self.prev = prev
        else:
            self.prev = None
        if next is not None:
            assert (isinstance(next, Node))
            self.next = next
        else:
            self.next = None
        if id is not None and (isinstance(id, int) or type(id) in [int32, uint32, int64, uint64]) and (id >= 0):
            self.id = id
        elif id is None:
            self.id = idgen.nextID(random.uniform(-180.0, 180.0), random.uniform(-90.0, 90.0), random.uniform(0., 75.0), 0.)
        else:
            self.id = None
        self.data = data

    def __deepcopy__(self, memodict={}):
        result = type(self)(prev=None, next=None, id=-1, data=None)
        result.registered = True
        result.id = self.id
        result.next = self.next
        result.prev = self.prev
        result.data = self.data
        return result

    def __del__(self):
        # print("Node.del() is called.")
        self.unlink()
        del self.data
        idgen.releaseID(self.id)

    def unlink(self):
        # print("Node.unlink() [id={}] is called.".format(self.id))
        if self.registered:
            if self.prev is not None:
                self.prev.set_next(self.next)
            if self.next is not None:
                self.next.set_prev(self.prev)
            self.registered = False
        self.prev = None
        self.next = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.next is None:
            raise StopIteration
        return self.next

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        if (self.data is not None) and (other.data is not None):
            return self.data == other.data
        else:
            return self.id == other.id

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        # print("less-than({} vs. {})".format(str(self),str(other)))
        if type(self) is not type(other):
            err_msg = "This object and the other object (type={}) do note have the same type.".format(str(type(other)))
            raise AttributeError(err_msg)
        return self.id < other.id

    def __le__(self, other):
        if type(self) is not type(other):
            err_msg = "This object and the other object (type={}) do note have the same type.".format(str(type(other)))
            raise AttributeError(err_msg)
        return self.id <= other.id

    def __gt__(self, other):
        if type(self) is not type(other):
            err_msg = "This object and the other object (type={}) do note have the same type.".format(str(type(other)))
            raise AttributeError(err_msg)
        return self.id > other.id

    def __ge__(self, other):
        if type(self) is not type(other):
            err_msg = "This object and the other object (type={}) do note have the same type.".format(str(type(other)))
            raise AttributeError(err_msg)
        return self.id >= other.id

    def __repr__(self):
        return '<%s.%s object at %s>' % (
            self.__class__.__module__,
            self.__class__.__name__,
            hex(id(self))
        )

    def __str__(self):
        return "Node(p: {}, n: {}, id: {}, d: {})".format(repr(self.prev), repr(self.next), self.id, repr(self.data))

    def __sizeof__(self):
        obj_size = sys.getsizeof(object)+sys.getsizeof(object)+sys.getsizeof(self.id)
        if self.data is not None:
            obj_size += sys.getsizeof(self.data)
        return obj_size

    def set_prev(self, prev):
        self.prev = prev

    def set_next(self, next):
        self.next = next

    def set_data(self, data):
        self.data = data

    def reset_data(self):
        self.data = None

    def reset_prev(self):
        self.prev = None

    def reset_next(self):
        self.prev = None

# global node_c_interface
node_c_interface = None
# global c_funcs
c_funcs = None


class NodeJIT(Node, ctypes.Structure):
    _fields_ = [('_c_prev_p', ctypes.c_void_p),
                ('_c_next_p', ctypes.c_void_p),
                ('_c_data_p', ctypes.c_void_p),
                ('_c_pu_affinity', ctypes.c_int)]

    init_node_c = None
    set_prev_ptr_c = None
    set_next_ptr_c = None
    set_data_ptr_c = None
    reset_prev_ptr_c = None
    reset_next_ptr_c = None
    reset_data_ptr_c = None

    def __init__(self, prev=None, next=None, id=None, data=None):
        super().__init__(prev=prev, next=next, id=id, data=data)
        if not c_lib_register.is_created("node") or not c_lib_register.is_compiled("node") or not c_lib_register.is_loaded("node"):
            c_lib_register.load("node", src_dir=os.path.dirname(os.path.abspath(__file__)))
        c_lib_register.register("node")
        self.registered = True
        global node_c_interface
        if node_c_interface is None:
            node_c_interface = c_lib_register.get("node")  # ["node"]

        func_params = NodeJIT_func_params()
        # func_params = []
        # func_params.append({"name": 'init_node', "return": None, "arguments": [ctypes.POINTER(NodeJIT)]})
        # func_params.append({"name": 'set_prev_ptr', "return": None, "arguments": [ctypes.POINTER(NodeJIT), ctypes.POINTER(NodeJIT)]})
        # func_params.append({"name": 'set_next_ptr', "return": None, "arguments": [ctypes.POINTER(NodeJIT), ctypes.POINTER(NodeJIT)]})
        # func_params.append({"name": 'set_data_ptr', "return": None, "arguments": [ctypes.POINTER(NodeJIT), ctypes.c_void_p]})
        # func_params.append({"name": 'set_pu_affinity', "return": None, "arguments": [ctypes.POINTER(NodeJIT), ctypes.c_int]})
        # func_params.append({"name": 'get_pu_affinity', "return": ctypes.c_int, "arguments": [ctypes.POINTER(NodeJIT)]})
        # func_params.append({"name": 'reset_prev_ptr', "return": None, "arguments": [ctypes.POINTER(NodeJIT)]})
        # func_params.append({"name": 'reset_next_ptr', "return": None, "arguments": [ctypes.POINTER(NodeJIT)]})
        # func_params.append({"name": 'reset_data_ptr', "return": None, "arguments": [ctypes.POINTER(NodeJIT)]})
        # func_params.append({"name": 'reset_pu_affinity', "return": None, "arguments": [ctypes.POINTER(NodeJIT)]})

        # -- Question: do we REALLY need to look them up all the time ? We count store that library link and just re-use it for each particle -- #
        global c_funcs
        if c_funcs is None:
            c_funcs = node_c_interface.load_functions(func_params)
        self.link_c_functions(c_funcs)
        # self.init_node_c = c_funcs['init_node']
        # self.set_prev_ptr_c = c_funcs['set_prev_ptr']
        # self.set_next_ptr_c = c_funcs['set_next_ptr']
        # self.set_data_ptr_c = c_funcs['set_data_ptr']
        # self.reset_prev_ptr_c = c_funcs['reset_prev_ptr']
        # self.reset_next_ptr_c = c_funcs['reset_next_ptr']
        # self.reset_data_ptr_c = c_funcs['reset_data_ptr']

        self.init_node_c(self)

        if self.prev is not None and isinstance(self.prev, NodeJIT):
            self.set_prev_ptr_c(self, self.prev)
        else:
            self.reset_prev_ptr_c(self)
        # self._c_self_p = ctypes.cast(self, ctypes.c_void_p)
        if self.next is not None and isinstance(self.next, NodeJIT):
            self.set_next_ptr_c(self, self.next)
        else:
            self.reset_next_ptr_c(self)

        if self.data is not None:   # and isinstance(ctypes.c_void_p):
            # self._c_data_p = ctypes.cast(self.data, ctypes.c_void_p)
            try:
                self.set_data_ptr_c(self, self.data.cdata())
            except AttributeError:
                self.set_data_ptr_c(self, ctypes.cast(self.data, ctypes.c_void_p))
        else:
            self.reset_data_ptr_c(self)

    def __deepcopy__(self, memodict={}):
        result = type(self)(prev=None, next=None, id=-1, data=None)
        result.id = self.id
        result.next = self.next
        result.prev = self.prev
        result.data = self.data
        c_lib_register.register("node")   # should rather be done by 'result' internally
        result.registered = True
        result.init_node_c = self.init_node_c
        result.set_prev_ptr_c = self.set_prev_ptr_c
        result.set_next_ptr_c = self.set_next_ptr_c
        result.set_data_ptr_c = self.set_data_ptr_c
        result.reset_prev_ptr_c = self.reset_prev_ptr_c
        result.reset_next_ptr_c = self.reset_next_ptr_c
        result.reset_data_ptr_c = self.reset_data_ptr_c
        result.init_node_c(self)
        if result.prev is not None and isinstance(result.prev, NodeJIT):
            result.set_prev_ptr_c(result, result.prev)
        else:
            result.reset_prev_ptr_c(result)
        if result.next is not None and isinstance(result.next, NodeJIT):
            result.set_next_ptr_c(result, result.next)
        else:
            result.reset_next_ptr_c(result)

        if result.data is not None:
            result.set_data_ptr_c(result, ctypes.cast(result.data, ctypes.c_void_p))
        else:
            result.reset_data_ptr_c(result)
        return result

    def __del__(self):
        # print("NodeJIT.del() [id={}] is called.".format(self.id))
        self.unlink()
        del self.data
        idgen.releaseID(self.id)

    def unlink(self):
        # print("NodeJIT.unlink() [id={}] is called.".format(self.id))
        if self.registered:
            if self.prev is not None:
                if self.next is not None:
                    self.prev.set_next(self.next)
                else:
                    # self.reset_next_ptr_c(self.prev)
                    self.prev.reset_next()
            if self.next is not None:
                if self.prev is not None:
                    self.next.set_prev(self.prev)
                else:
                    # self.reset_prev_ptr_c(self.next)
                    self.next.reset_prev()
            self.reset_prev_ptr_c(self)
            self.reset_next_ptr_c(self)
            self.reset_data_ptr_c(self)
            c_lib_register.deregister("node")
            self.registered = False
        self.prev = None
        self.next = None

    def __repr__(self):
        return super().__repr__()

    def __str__(self):
        return super().__str__()

    def __sizeof__(self):
        return super().__sizeof__()+sys.getsizeof(self._fields_)

    def __eq__(self, other):
        return super().__eq__(other)

    def __ne__(self, other):
        return super().__ne__(other)

    def __lt__(self, other):
        return super().__lt__(other)

    def __le__(self, other):
        return super().__le__(other)

    def __gt__(self, other):
        return super().__gt__(other)

    # def __ge__(self, other):
    #     return super().__ge__(other)

    def link_c_functions(self, c_func_dict):
        self.init_node_c = c_func_dict['init_node']
        self.set_prev_ptr_c = c_func_dict['set_prev_ptr']
        self.set_next_ptr_c = c_func_dict['set_next_ptr']
        self.set_data_ptr_c = c_func_dict['set_data_ptr']
        self.reset_prev_ptr_c = c_func_dict['reset_prev_ptr']
        self.reset_next_ptr_c = c_func_dict['reset_next_ptr']
        self.reset_data_ptr_c = c_func_dict['reset_data_ptr']

    def set_data(self, data):
        super().set_data(data)
        if self.registered:
            self.update_data()

    def set_prev(self, prev):
        super().set_prev(prev)
        if self.registered:
            self.update_prev()

    def set_next(self, next):
        super().set_next(next)
        if self.registered:
            self.update_next()

    def reset_data(self):
        super().reset_data()
        if self.registered:
            self.reset_data_ptr_c(self)

    def reset_prev(self):
        super().reset_prev()
        if self.registered:
            self.reset_prev_ptr_c(self)

    def reset_next(self):
        super().reset_next()
        if self.registered:
            self.reset_next_ptr_c(self)

    def update_prev(self):
        if self.prev is not None and isinstance(self.prev, NodeJIT):
            # self._c_prev_p = ctypes.cast(self.prev, ctypes.c_void_p)
            # self._c_prev_p = self.prev._c_self_p
            self.set_prev_ptr_c(self, self.prev)
        else:
            self.reset_prev_ptr_c(self)

    def update_next(self):
        if self.next is not None and isinstance(self.next, NodeJIT):
            # self._c_next_p = ctypes.cast(self.next, ctypes.c_void_p)
            # self._c_next_p = self.next._c_self_p
            self.set_next_ptr_c(self, self.next)
        else:
            self.reset_next_ptr_c(self)

    def update_data(self):
        if self.data is not None:   # and isinstance(ctypes.c_void_p):
            # self._c_data_p = ctypes.cast(self.data, ctypes.c_void_p)
            try:
                self.set_data_ptr_c(self, self.data.cdata())
            except AttributeError:
                self.set_data_ptr_c(self, ctypes.cast(self.data, ctypes.c_void_p))
        else:
            self.reset_data_ptr_c(self)


def NodeJIT_func_params():  #node_c_func_params =
    return [{"name": 'init_node', "return": None, "arguments": [ctypes.POINTER(NodeJIT)]},
            {"name": 'set_prev_ptr', "return": None, "arguments": [ctypes.POINTER(NodeJIT), ctypes.POINTER(NodeJIT)]},
            {"name": 'set_next_ptr', "return": None, "arguments": [ctypes.POINTER(NodeJIT), ctypes.POINTER(NodeJIT)]},
            {"name": 'set_next_ptr', "return": None, "arguments": [ctypes.POINTER(NodeJIT), ctypes.POINTER(NodeJIT)]},
            {"name": 'set_data_ptr', "return": None, "arguments": [ctypes.POINTER(NodeJIT), ctypes.c_void_p]},
            {"name": 'set_pu_affinity', "return": None, "arguments": [ctypes.POINTER(NodeJIT), ctypes.c_int]},
            {"name": 'get_pu_affinity', "return": ctypes.c_int, "arguments": [ctypes.POINTER(NodeJIT)]},
            {"name": 'reset_prev_ptr', "return": None, "arguments": [ctypes.POINTER(NodeJIT)]},
            {"name": 'reset_next_ptr', "return": None, "arguments": [ctypes.POINTER(NodeJIT)]},
            {"name": 'reset_data_ptr', "return": None, "arguments": [ctypes.POINTER(NodeJIT)]},
            {"name": 'reset_pu_affinity', "return": None, "arguments": [ctypes.POINTER(NodeJIT)]}]



