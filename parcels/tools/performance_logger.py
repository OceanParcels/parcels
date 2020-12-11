import time as time_module
import numpy as np

try:
    from mpi4py import MPI
except:
    MPI = None

from threading import Thread
from threading import Event
from time import sleep

class TimingLog():
    stime = 0
    etime = 0
    mtime = 0
    _samples = []
    _times_steps = []
    _iter = 0

    def __init__(self):
        self.stime = 0
        self.etime = 0
        self.mtime = 0
        self._samples = []
        self._times_steps = []
        self._iter = 0

    @property
    def timing(self):
        return self._times_steps

    @property
    def samples(self):
        return self._samples

    def __len__(self):
        return len(self._samples)

    def get_values(self):
        return self._times_steps

    def get_value(self, index):
        N = len(self._times_steps)
        result = 0
        if N > 0:
            result = self._times_steps[min(max(index, 0), N - 1)]
        return result

    def start_timing(self):
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            if mpi_rank == 0:
                # self.stime = MPI.Wtime()
                # self.stime = time_module.perf_counter()
                self.stime = time_module.process_time()
        else:
            self.stime = time_module.process_time()

    def stop_timing(self):
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            if mpi_rank == 0:
                # self.etime = MPI.Wtime()
                # self.etime = time_module.perf_counter()
                self.etime = time_module.process_time()
        else:
            self.etime = time_module.process_time()

    def accumulate_timing(self):
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            if mpi_rank == 0:
                self.mtime += (self.etime-self.stime)
            else:
                self.mtime = 0
        else:
            self.mtime += (self.etime-self.stime)

    def advance_iteration(self):
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            if mpi_rank == 0:
                self._times_steps.append(self.mtime)
                self._samples.append(self._iter)
                self._iter += 1
            self.mtime = 0
        else:
            self._times_steps.append(self.mtime)
            self._samples.append(self._iter)
            self._iter += 1
            self.mtime = 0

    def add_aux_measure(self, value):
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            if mpi_rank == 0:
                self.mtime += value
            else:
                self.mtime += 0
        else:
            self.mtime += value

    def sum(self):
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            if mpi_rank == 0:
                result = np.array(self._times_steps).sum()
                return result
            else:
                return 0
        else:
            result = np.array(self._times_steps).sum()

    def reset(self):
        if self._times_steps is not None and len(self._times_steps) > 0:
            del self._times_steps[:]
        if self._samples is not None and len(self._samples) > 0:
            del self._samples[:]
        self.stime = 0
        self.etime = 0
        self.mtime = 0
        self._samples = []
        self._times_steps = []
        self._iter = 0


class ParamLogging():
    _samples = []
    _params = []
    _iter = 0

    def __init__(self):
        self._samples = []
        self._params = []
        self._iter = 0

    @property
    def samples(self):
        return self._samples

    @property
    def params(self):
        return self._params

    def get_params(self):
        return self._params

    def get_param(self, index):
        N = len(self._params)
        result = 0
        if N > 0:
            result = self._params[min(max(index, 0), N-1)]
        return result

    def __len__(self):
        return len(self._samples)

    def advance_iteration(self, param):
        if MPI:
            # mpi_comm = MPI.COMM_WORLD
            # mpi_rank = mpi_comm.Get_rank()

            self._params.append(param)
            self._samples.append(self._iter)
            self._iter += 1
            # if mpi_rank == 0:
            #     self._params.append(param)
            #     self._samples.append(self._iter)
            #     self._iter += 1
        else:
            self._params.append(param)
            self._samples.append(self._iter)
            self._iter += 1

class Asynchronous_ParamLogging():
    _samples = []
    _params = []
    _iter = 0
    _measure_func = None
    _measure_start_value = None  # for differential measurements
    _measure_partial_values = []
    _measure_interval = 0.25  # 250 ms
    _event = None
    _thread = None
    differential_measurement = False

    def __init__(self):
        self._samples = []
        self._params = []
        self._measure_partial_values = []
        self._iter = 0
        self._measure_func = None
        self._measure_start_value = None
        self._measure_interval = 0.25  # 250 ms
        self._event = None
        self._thread = None
        self.differential_measurement = False

    def __del__(self):
        del self._samples[:]
        del self._params[:]
        del self._measure_partial_values[:]

    @property
    def samples(self):
        return self._samples

    @property
    def params(self):
        return self._params

    @property
    def measure_func(self):
        return self._measure_func

    @measure_func.setter
    def measure_func(self, function):
        self._measure_func = function

    @property
    def measure_interval(self):
        return self._measure_interval

    @measure_interval.setter
    def measure_interval(self, interval):
        """
        Set measure interval in seconds
        :param interval: interval in seconds (fractional possible)
        :return: None
        """
        self._measure_interval = interval

    @property
    def measure_start_value(self):
        return self._measure_start_value

    @measure_start_value.setter
    def measure_start_value(self, value):
        self._measure_start_value = value

    def async_run(self):
        if self.differential_measurement:
            self.async_run_diff_measurement()
        else:
            pass

    def async_run_diff_measurement(self):
        if self._measure_start_value is None:
            self._measure_start_value = self._measure_func()
            self._measure_partial_values.append(0)
        while not self._event.is_set():
            self._measure_partial_values.append( self._measure_func()-self._measure_start_value )
            sleep(self._measure_interval)

    def async_run_measurement(self):
        while not self._event.is_set():
            self._measure_partial_values.append( self.measure_func() )
            sleep(self.measure_interval)

    def start_partial_measurement(self):
        assert self._measure_func is not None, "Measurement function is None - invalid. Exiting ..."
        assert self._thread is None, "Measurement already running - double-start invalid. Exiting ..."
        if len(self._measure_partial_values) > 0:
            del self._measure_partial_values[:]
            self._measure_partial_values = []
        self._event = Event()
        self._thread = Thread(target=self.async_run_measurement)
        self._thread.start()

    def stop_partial_measurement(self):
        """
        function to stop the measurement. The function also internally advances the iteration with the mean (or max)
        of the measured partial values.
        :return: None
        """
        self._event.set()
        self._thread.join()
        sleep(self._measure_interval)
        del self._thread
        self._thread = None
        self._measure_start_value = None
        # param_partial_mean = np.array(self._measure_partial_values).mean()
        param_partial_mean = np.array(self._measure_partial_values).max()
        self.advance_iteration(param_partial_mean)

    def get_params(self):
        return self._params

    def get_param(self, index):
        N = len(self._params)
        result = 0
        if N > 0:
            result = self._params[min(max(index, 0), N-1)]
        return result

    def __len__(self):
        return len(self._samples)

    def advance_iteration(self, param):
        if MPI:
            # mpi_comm = MPI.COMM_WORLD
            # mpi_rank = mpi_comm.Get_rank()

            self._params.append(param)
            self._samples.append(self._iter)
            self._iter += 1
            # if mpi_rank == 0:
            #     self._params.append(param)
            #     self._samples.append(self._iter)
            #     self._iter += 1
        else:
            self._params.append(param)
            self._samples.append(self._iter)
            self._iter += 1

