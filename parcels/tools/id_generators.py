import random   # could be python's random if parcels not active; can be parcel's random; can be numpy's random
from abc import ABC, abstractmethod
# from numpy import random as nprandom
# from multiprocessing import Process
from threading import Thread
from .message_service import mpi_execute_requested_messages as executor
# from os import getpid
import numpy as np

try:
    from mpi4py import MPI
except:
    MPI = None


class BaseIdGenerator(ABC):
    _total_ids = 0
    _recover_ids = False

    def __init__(self):
        self._total_ids = 0

    def setTimeLine(self, min_time, max_time):
        pass

    def setDepthLimits(self, min_depth, max_depth):
        pass

    def preGenerateIDs(self, high_value):
        pass

    def permuteIDs(self):
        pass

    def close(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @property
    def total_length(self):
        return self._total_ids

    @property
    def recover_ids(self):
        return self._recover_ids

    @recover_ids.setter
    def recover_ids(self, bool_param):
        self._recover_ids = bool_param

    def enable_ID_recovery(self):
        self._recover_ids = True

    def disable_ID_recovery(self):
        self._recover_ids = False

    @abstractmethod
    def getID(self, lon, lat, depth, time):
        pass

    def nextID(self, lon, lat, depth, time):
        return self.getID(lon, lat, depth, time)

    @abstractmethod
    def releaseID(self, id):
        pass

    @abstractmethod
    def get_length(self):
        return self.__len__()

    @abstractmethod
    def get_total_length(self):
        return self._total_ids


class SequentialIdGenerator(BaseIdGenerator):
    released_ids = []
    next_id = 0

    def __init__(self):
        super(SequentialIdGenerator, self).__init__()
        self.released_ids = []
        self.next_id = np.uint64(0)
        self._recover_ids = False

    def __del__(self):
        if len(self.released_ids) > 0:
            del self.released_ids

    def getID(self, lon, lat, depth, time):
        n = len(self.released_ids)
        if n == 0:
            result = self.next_id
            self.next_id += 1
            self._total_ids += 1
            return np.uint64(result)
        else:
            result = self.released_ids.pop(n-1)
            return np.uint64(result)

    def releaseID(self, id):
        if not self._recover_ids:
            return
        self.released_ids.append(id)

    def preGenerateIDs(self, high_value):
        if len(self.released_ids) > 0:
            self.released_ids.clear()
        self.released_ids = [i for i in range(0, high_value)]
        self.next_id = high_value

    def permuteIDs(self):
        n = len(self.released_ids)
        indices = random.randint(0, n, 2*n)
        for index in indices:
            id = self.released_ids.pop(index)
            self.released_ids.append(id)

    def __len__(self):
        return self.next_id

    def get_length(self):
        return self.__len__()

    def get_total_length(self):
        return self._total_ids


class SpatioTemporalIdGenerator(BaseIdGenerator):
    """Generates 64-bit IDs"""
    timebounds = np.zeros(2, dtype=np.float64)
    depthbounds = np.zeros(2, dtype=np.float32)
    local_ids = None
    released_ids = {}
    _total_ids = 0

    def __init__(self):
        super(SpatioTemporalIdGenerator, self).__init__()
        self.timebounds = np.zeros(2, dtype=np.float64)
        self.depthbounds = np.zeros(2, dtype=np.float32)
        self.local_ids = np.zeros((360, 180, 128, 256), dtype=np.uint32)
        self.released_ids = {}  # 32-bit spatio-temporal index => []
        self._total_ids = 0
        self._recover_ids = False

    def __del__(self):
        # occupancy_file = path.join( get_cache_dir(), 'id_occupancy.npy')
        # idreleases_file = path.join( get_cache_dir(), 'id_releases.pkl' )
        # if path.exists(occupancy_file):
        #     remove(occupancy_file)
        # if path.exists(idreleases_file):
        #     remove(idreleases_file)
        if self.local_ids is not None:
            del self.local_ids
        if len(self.released_ids) > 0:
            del self.released_ids

    def setTimeLine(self, min_time=0.0, max_time=1.0):
        self.timebounds = np.array([min_time, max_time], dtype=np.float64)

    def setDepthLimits(self, min_depth=0.0, max_depth=1.0):
        self.depthbounds = np.array([min_depth, max_depth], dtype=np.float32)

    def getID(self, lon, lat, depth, time):
        if lon < -180.0 or lon > 180.0:
            vsgn = np.sign(lon)
            lon = np.fmod(np.fabs(lon), 180.0) * vsgn
        if lat < -90.0 or lat > 90.0:
            vsgn = np.sign(lat)
            lat = np.fmod(np.fabs(lat), 90.0) * vsgn
        if depth is None:
            depth = self.depthbounds[0]
        if time is None:
            time = self.timebounds[0]
        lon_discrete = np.int32(lon)
        lat_discrete = np.int32(lat)
        depth_discrete = (depth-self.depthbounds[0])/(self.depthbounds[1]-self.depthbounds[0])
        depth_discrete = np.int32(127.0 * depth_discrete)
        time_discrete = (time-self.timebounds[0])/(self.timebounds[1]-self.timebounds[0])
        time_discrete = np.int32(255.0 * time_discrete)
        lon_index = np.uint32(np.int32(lon_discrete)+180)
        lat_index = np.uint32(np.int32(lat_discrete)+90)
        depth_index = np.uint32(np.int32(depth_discrete))
        time_index = np.uint32(np.int32(time_discrete))
        # if MPI:
        #     id = self._distribute_next_id_by_file(lon_index, lat_index, depth_index, time_index)
        # else:
        #     id = self._get_next_id(lon_index, lat_index, depth_index, time_index)
        id = self._get_next_id(lon_index, lat_index, depth_index, time_index)
        return id

    def nextID(self, lon, lat, depth, time):
        return self.getID(lon, lat, depth, time)

    def releaseID(self, id):
        full_bits = np.uint32(4294967295)
        nil_bits = np.int32(0)
        spatiotemporal_id = np.bitwise_and(np.bitwise_or(np.left_shift(np.int64(full_bits), 32), np.int64(nil_bits)), np.int64(id))
        spatiotemporal_id = np.uint32(np.right_shift(spatiotemporal_id, 32))
        local_id = np.bitwise_and(np.bitwise_or(np.left_shift(np.int64(nil_bits), 32), np.int64(full_bits)), np.int64(id))
        local_id = np.uint32(local_id)
        # if MPI:
        #     self._gather_released_ids_by_file(spatiotemporal_id, local_id)
        # else:
        #     self._release_id(spatiotemporal_id, local_id)
        self._release_id(spatiotemporal_id, local_id)

    def __len__(self):
        # if MPI:
        #     # return self._length_()
        #     return np.sum(self.local_ids) + sum([len(entity) for entity in self.released_ids])
        # else:
        #     return np.sum(self.local_ids) + sum([len(entity) for entity in self.released_ids])
        return np.sum(self.local_ids) + sum([len(entity) for entity in self.released_ids])

    @property
    def total_length(self):
        return self._total_ids

    def get_length(self):
        return self.__len__()

    def get_total_length(self):
        # return self._total_ids
        return self.total_length

    # def _distribute_next_id(self, lon_index, lat_index, depth_index, time_index):
    #     mpi_comm = MPI.COMM_WORLD
    #     mpi_rank = mpi_comm.Get_rank()
    #     mpi_size = mpi_comm.Get_size()
    #     snd_requested_id = np.zeros((mpi_size, 1), dtype=np.byte)
    #     rcv_requested_id = np.zeros((mpi_size, 1), dtype=np.byte)
    #     snd_requested_add = np.zeros((mpi_size, 4), dtype=np.uint32)
    #     rcv_requested_add = np.zeros((mpi_size, 4), dtype=np.uint32)
    #     return_id = np.zeros(mpi_size, dtype=np.uint64)
    #     return_id.fill(np.iinfo(np.uint64).max)
    #     snd_requested_id[mpi_rank] = 1
    #     snd_requested_add[mpi_rank, :] = np.array([lon_index, lat_index, depth_index, time_index], dtype=np.uint32)
    #     mpi_comm.Reduce(snd_requested_id, rcv_requested_id, op=MPI.MAX, root=0)
    #     # rcv_requested_id = mpi_comm.reduce(snd_requested_id, op=MPI.MAX, root=0)
    #     mpi_comm.Reduce(snd_requested_add, rcv_requested_add, op=MPI.MAX, root=0)
    #     if mpi_rank == 0:
    #         for i in range(mpi_size):
    #             if rcv_requested_id[i] > 0:
    #                 return_id[i] = self._get_next_id(rcv_requested_add[i, 0], rcv_requested_add[i, 1], rcv_requested_add[i, 2], rcv_requested_add[i, 3])
    #     # mpi_comm.Bcast(return_id, root=0)
    #     return_id = mpi_comm.bcast(return_id, root=0)
    #     return return_id[mpi_rank]

    # def _distribute_next_id_by_file(self, lon_index, lat_index, depth_index, time_index):
    #     # mpi_comm = MPI.COMM_WORLD
    #     # mpi_rank = mpi_comm.Get_rank()
    #     # mpi_size = mpi_comm.Get_size()

    #     return_id = None
    #     access_flag_file = path.join( get_cache_dir(), 'id_access' )
    #     occupancy_file = path.join( get_cache_dir(), 'id_occupancy.npy')
    #     idreleases_file = path.join( get_cache_dir(), 'id_releases.pkl' )
    #     while path.exists(access_flag_file):
    #         sleep(0.1)
    #     with open(access_flag_file, 'wb') as f_access:
    #         f_access.write(bytearray([True,]))
    #         # self.local_ids = np.fromfile(occupancy_file, dtype=np.uint32)
    #         self.local_ids = np.load(occupancy_file)
    #         with open(idreleases_file, 'rb') as f_idrel:
    #             self.released_ids = pickle.load( f_idrel )
    #         return_id = self._get_next_id(lon_index, lat_index, depth_index, time_index)
    #         with open(idreleases_file, 'wb') as f_idrel:
    #             pickle.dump(self.released_ids, f_idrel)
    #         # self.local_ids.tofile(occupancy_file)
    #         np.save(occupancy_file, self.local_ids)
    #     remove(access_flag_file)
    #
    #     return return_id

    # def _gather_released_ids(self, spatiotemporal_id, local_id):
    #     mpi_comm = MPI.COMM_WORLD
    #     mpi_rank = mpi_comm.Get_rank()
    #     mpi_size = mpi_comm.Get_size()
    #     snd_release_id = np.zeros(mpi_size, dtype=np.byte)
    #     rcv_release_id = np.zeros(mpi_size, dtype=np.byte)
    #     snd_release_add = np.zeros((mpi_size, 2), dtype=np.uint32)
    #     rcv_release_add = np.zeros((mpi_size, 2), dtype=np.uint32)
    #     snd_release_id[mpi_rank] = 1
    #     snd_release_add[mpi_rank, :] = np.array([spatiotemporal_id, local_id], dtype=np.uint32)
    #     mpi_comm.Reduce(snd_release_id, rcv_release_id, op=MPI.MAX, root=0)
    #     mpi_comm.Reduce(snd_release_add, rcv_release_add, op=MPI.MAX, root=0)
    #     if mpi_rank == 0:
    #         for i in range(mpi_size):
    #             if rcv_release_id[i] > 0:
    #                 self._release_id(rcv_release_add[i, 0], rcv_release_add[i, 1])

    # def _gather_released_ids_by_file(self, spatiotemporal_id, local_id):
    #
    #     return_id = None
    #     access_flag_file = path.join( get_cache_dir(), 'id_access' )
    #     occupancy_file = path.join( get_cache_dir(), 'id_occupancy.npy')
    #     idreleases_file = path.join( get_cache_dir(), 'id_releases.pkl' )
    #     while path.exists(access_flag_file):
    #         sleep(0.1)
    #     with open(access_flag_file, 'wb') as f_access:
    #         f_access.write(bytearray([True,]))
    #         # self.local_ids = np.fromfile(occupancy_file, dtype=np.uint32)
    #         self.local_ids = np.load(occupancy_file)
    #         with open(idreleases_file, 'rb') as f_idrel:
    #             self.released_ids = pickle.load( f_idrel )
    #         self._release_id(spatiotemporal_id, local_id)
    #         with open(idreleases_file, 'wb') as f_idrel:
    #             pickle.dump(self.released_ids, f_idrel)
    #         # self.local_ids.tofile(occupancy_file)
    #         np.save(occupancy_file, self.local_ids)
    #     remove(access_flag_file)
    #
    #     return return_id

    def _get_next_id(self, lon_index, lat_index, depth_index, time_index):
        local_index = -1
        id = np.left_shift(lon_index, 23) + np.left_shift(lat_index, 15) + np.left_shift(depth_index, 8) + time_index
        # id = np.left_shift(lon_index, 23) + np.left_shift(lat_index, 15) + np.left_shift(depth_index, 8) + time
        # == print("requtested indices: ({}, {}, {}, {})".format(lon_index, lat_index, depth_index, time_index)) == #
        # == print("spatial id: {}".format(id)) == #
        if len(self.released_ids) > 0 and (id in self.released_ids.keys()) and len(self.released_ids[id]) > 0:
            # mlist = self.released_ids[id]
            local_index = np.uint32(self.released_ids[id].pop())
            if len(self.released_ids[id]) <= 0:
                del self.released_ids[id]
        else:
            local_index = self.local_ids[lon_index, lat_index, depth_index, time_index]
            self.local_ids[lon_index, lat_index, depth_index, time_index] += 1
        id = np.int64(id)
        id = np.bitwise_or(np.left_shift(id, 32), np.int64(local_index))
        id = np.uint64(id)
        self._total_ids += 1
        return id

    def _release_id(self, spatiotemporal_id, local_id):
        if not self._recover_ids:
            return
        if spatiotemporal_id not in self.released_ids.keys():
            self.released_ids[spatiotemporal_id] = []
        self.released_ids[spatiotemporal_id].append(local_id)


# ====== ====== ====== ====== #
# == Imports and codes for == #
# == asyncronous ID gen.   == #
# == via Threads/Processes == #
# ====== ====== ====== ====== #
# #from multiprocessing import Process, Pipe
# from threading import Thread
# from multiprocessing import Process
# # from multiprocessing.connection import Connection
# from .message_service import mpi_execute_requested_messages as executor
# from sys import stdout
# from os import getpid
# from parcels.tools import logger
#
# class GenerateID_Service(object):
#     _request_tag = 5
#     _response_tag = 6
#
#     def __init__(self, base_generator_obj):
#         super(GenerateID_Service, self).__init__()
#         self._service_process = None
#         self._serverrank = 0
#         self._request_tag = 5
#         self._response_tag = 6
#         self._use_subprocess = True
#
#         if MPI:
#             mpi_comm = MPI.COMM_WORLD
#             mpi_rank = mpi_comm.Get_rank()
#             mpi_size = mpi_comm.Get_size()
#             if mpi_size <= 1:
#                 self._use_subprocess = False
#             else:
#                 self._serverrank = mpi_size-1
#                 if mpi_rank == self._serverrank:
#                     self._service_process = Thread(target=executor, name="IdService", args=(base_generator_obj, self._request_tag, self._response_tag), daemon=True)
#                     self._service_process.start()
#                 self._subscribe_()
#         else:
#             self._use_subprocess = False
#
#         if not self._use_subprocess:
#             self._service_process = base_generator_obj()
#
#     def __del__(self):
#         self._abort_()
#
#     def _subscribe_(self):
#         if MPI and self._use_subprocess:
#             mpi_comm = MPI.COMM_WORLD
#             mpi_rank = mpi_comm.Get_rank()
#             data_package = {}
#             data_package["func_name"] = "thread_subscribe"
#             data_package["args"] = 0
#             data_package["src_rank"] = mpi_rank
#             mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
#
#     def _abort_(self):
#         if MPI and self._use_subprocess:
#             mpi_comm = MPI.COMM_WORLD
#             mpi_rank = mpi_comm.Get_rank()
#             data_package = {}
#             data_package["func_name"] = "thread_abort"
#             data_package["args"] = 0
#             data_package["src_rank"] = mpi_rank
#             mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
#
#     def close(self):
#         self._abort_()
#
#     def setTimeLine(self, min_time=0.0, max_time=1.0):
#         if MPI and self._use_subprocess:
#             mpi_comm = MPI.COMM_WORLD
#             mpi_rank = mpi_comm.Get_rank()
#             if mpi_rank == 0:
#                 data_package = {}
#                 data_package["func_name"] = "setTimeLine"
#                 data_package["args"] = 2
#                 data_package["argv"] = [min_time, max_time]
#                 data_package["src_rank"] = mpi_rank
#                 mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
#         else:
#             self._service_process.setTimeLine(min_time, max_time)
#
#     def setDepthLimits(self, min_depth=0.0, max_depth=1.0):
#         if MPI and self._use_subprocess:
#             mpi_comm = MPI.COMM_WORLD
#             mpi_rank = mpi_comm.Get_rank()
#             if mpi_rank == 0:
#                 data_package = {}
#                 data_package["func_name"] = "setDepthLimits"
#                 data_package["args"] = 2
#                 data_package["argv"] = [min_depth, max_depth]
#                 data_package["src_rank"] = mpi_rank
#                 mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
#         else:
#             self._service_process.setDepthLimits(min_depth, max_depth)
#
#     def getID(self, lon, lat, depth, time):
#         if MPI and self._use_subprocess:
#             mpi_comm = MPI.COMM_WORLD
#             mpi_rank = mpi_comm.Get_rank()
#
#             data_package = {}
#             data_package["func_name"] = "getID"
#             data_package["args"] = 4
#             data_package["argv"] = [lon, lat, depth, time]
#             data_package["src_rank"] = mpi_rank
#             mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
#             data = mpi_comm.recv(source=self._serverrank, tag=self._response_tag)
#             return int(data["result"])
#         else:
#             return self._service_process.getID(lon, lat, depth, time)
#
#     def nextID(self, lon, lat, depth, time):
#         return self.getID(lon, lat, depth, time)
#
#     def releaseID(self, id):
#         if MPI and self._use_subprocess:
#             mpi_comm = MPI.COMM_WORLD
#             mpi_rank = mpi_comm.Get_rank()
#
#             data_package = {}
#             data_package["func_name"] = "releaseID"
#             data_package["args"] = 1
#             data_package["argv"] = [id, ]
#             data_package["src_rank"] = mpi_rank
#             mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
#         else:
#             self._service_process.releaseID(id)
#
#     def get_length(self):
#         if MPI and self._use_subprocess:
#             mpi_comm = MPI.COMM_WORLD
#             mpi_rank = mpi_comm.Get_rank()
#
#             data_package = {}
#             data_package["func_name"] = "get_length"
#             data_package["args"] = 0
#             data_package["src_rank"] = mpi_rank
#             mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
#             data = mpi_comm.recv(source=self._serverrank, tag=self._response_tag)
#
#             return int(data["result"])
#         else:
#             return self._service_process.__len__()
#
#     def get_total_length(self):
#         if MPI and self._use_subprocess:
#             mpi_comm = MPI.COMM_WORLD
#             mpi_rank = mpi_comm.Get_rank()
#
#             data_package = {}
#             data_package["func_name"] = "get_total_length"
#             data_package["args"] = 0
#             data_package["src_rank"] = mpi_rank
#             mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
#             data = mpi_comm.recv(source=self._serverrank, tag=self._response_tag)
#
#             return int(data["result"])
#         else:
#             return self._service_process.total_length
#
#     def __len__(self):
#         return self.get_length()
#
#     @property
#     def total_length(self):
#         return self.get_total_length()



class GenerateID_Service(BaseIdGenerator):
    _request_tag = 5
    _response_tag = 6

    def __init__(self, base_generator_obj):
        super(GenerateID_Service, self).__init__()
        self._service_process = None
        self._serverrank = 0
        self._request_tag = 5
        self._response_tag = 6
        self._recover_ids = False
        self._use_subprocess = True

        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            mpi_size = mpi_comm.Get_size()
            if mpi_size <= 1:
                self._use_subprocess = False
            else:
                self._serverrank = mpi_size-1
                if mpi_rank == self._serverrank:
                    # self._service_process = Process(target=executor, name="IdService", args=(service_bundle, base_generator_obj), daemon=True)
                    # self._service_process.start()
                    # print("Starting ID service process")
                    # logger.info("Starting ID service process")
                    self._service_process = Thread(target=executor, name="IdService", args=(base_generator_obj, self._request_tag, self._response_tag), daemon=True)
                    # self._service_process.daemon = True
                    self._service_process.start()
                    # executor(base_generator_obj, self._request_tag, self._response_tag)
                # mpi_comm.Barrier()
                # logger.info("worker - MPI rank: {} pid: {}".format(mpi_rank, getpid()))
                self._subscribe_()
        else:
            self._use_subprocess = False

        if not self._use_subprocess:
            self._service_process = base_generator_obj()

    def __del__(self):
        self._abort_()

    def _subscribe_(self):
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            data_package = {}
            data_package["func_name"] = "thread_subscribe"
            data_package["args"] = 0
            data_package["src_rank"] = mpi_rank
            mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)

    def _abort_(self):
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            data_package = {}
            data_package["func_name"] = "thread_abort"
            data_package["args"] = 0
            data_package["src_rank"] = mpi_rank
            mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)

    def close(self):
        self._abort_()

    def setTimeLine(self, min_time=0.0, max_time=1.0):
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            if mpi_rank == 0:
                data_package = {}
                data_package["func_name"] = "setTimeLine"
                data_package["args"] = 2
                data_package["argv"] = [min_time, max_time]
                data_package["src_rank"] = mpi_rank
                mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
        else:
            self._service_process.setTimeLine(min_time, max_time)

    def setDepthLimits(self, min_depth=0.0, max_depth=1.0):
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            if mpi_rank == 0:
                data_package = {}
                data_package["func_name"] = "setDepthLimits"
                data_package["args"] = 2
                data_package["argv"] = [min_depth, max_depth]
                data_package["src_rank"] = mpi_rank
                mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
        else:
            self._service_process.setDepthLimits(min_depth, max_depth)

    def getID(self, lon, lat, depth, time):
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()

            data_package = {}
            data_package["func_name"] = "getID"
            data_package["args"] = 4
            data_package["argv"] = [lon, lat, depth, time]
            data_package["src_rank"] = mpi_rank
            mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
            data = mpi_comm.recv(source=self._serverrank, tag=self._response_tag)
            return int(data["result"])
        else:
            return self._service_process.getID(lon, lat, depth, time)

    def nextID(self, lon, lat, depth, time):
        return self.getID(lon, lat, depth, time)

    def releaseID(self, id):
        if not self._recover_ids:
            return
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()

            data_package = {}
            data_package["func_name"] = "releaseID"
            data_package["args"] = 1
            data_package["argv"] = [id, ]
            data_package["src_rank"] = mpi_rank
            mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
        else:
            self._service_process.releaseID(id)

    def get_length(self):
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()

            data_package = {}
            data_package["func_name"] = "get_length"
            data_package["args"] = 0
            data_package["src_rank"] = mpi_rank
            mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
            data = mpi_comm.recv(source=self._serverrank, tag=self._response_tag)

            return int(data["result"])
        else:
            return self._service_process.__len__()

    def get_total_length(self):
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()

            data_package = {}
            data_package["func_name"] = "get_total_length"
            data_package["args"] = 0
            data_package["src_rank"] = mpi_rank
            mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
            data = mpi_comm.recv(source=self._serverrank, tag=self._response_tag)

            return int(data["result"])
        else:
            return self._service_process.total_length

    def __len__(self):
        return self.get_length()

    @property
    def total_length(self):
        return self.get_total_length()
