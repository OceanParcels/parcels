# from os import getpid
# from parcels.tools import logger


try:
    from mpi4py import MPI
except:
    MPI = None


def mpi_execute_requested_messages(exec_class, request_tag=0, response_tag=1):
    """
    A sub-thread/sub-process main function that manages a central (i.e. global) object.
    The process is that MPI processes can subscribe (via 'thread_subscribe' as function name) to the message queue. Then,
    those processes can send point-to-point message requests for functions of the central object to be executed.
    Potential results values are returned via message.
    IMPORTANT: upon MPI execution end, each process needs to send an 'abort' signal ('thread_abort' as function name) to
    unsubscribe from the message queue so to leave the message queue in an orderly fashion.
    """
    requester_obj = exec_class()
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    # logger.info("service - MPI rank: {} pid: {}".format(mpi_rank, getpid()))
    _subscribed = {}
    _running = True
    while _running:
        msg_status = MPI.Status()
        msg = mpi_comm.irecv(source=MPI.ANY_SOURCE, tag=request_tag)
        test_result = msg.test(status=msg_status)
        # while not (isinstance(test_result, tuple) or isinstance(test_result, list)) or ((test_result[0] == False) or (test_result[0] == True and not isinstance(test_result[1], dict))):
        # while (test_result[0] == False) or (test_result[0] == True and not isinstance(test_result[1], dict)):
        while (not test_result[0]) or (test_result[0] and not isinstance(test_result[1], dict)):
            test_result = msg.test(status=msg_status)

        request_package = test_result[1]
        # logger.info("ID serv. - recv.: {} - (srv. rank: {}; snd. rank: {}; pkg. rank: {}".format(request_package["func_name"], mpi_rank, msg_status.Get_source(), request_package["src_rank"]))
        # logger.info("package received: {}".format(request_package))
        assert isinstance(request_package, dict)
        # dst = int(request_package["src_rank"])

        # logger.info("Package: {}".format(request_package))
        func_name = request_package["func_name"]
        args = int(request_package["args"])
        argv = []
        if args > 0:
            argv = request_package["argv"]

        if func_name == "thread_subscribe":
            # logger.info("'subscribe' message received.")
            _subscribed[msg_status.Get_source()] = True
        elif func_name == "thread_abort":
            # logger.info("'abort' message received (src: {}).".format(msg_status.Get_source()))
            _subscribed[msg_status.Get_source()] = False
            # logger.info("Subscribers: {}".format( _subscribed ))
            _running = False
            for flag in _subscribed:
                _running |= flag
            # logger.warn("Subscribed: {}".format(_subscribed))
            if not _running:
                break
        else:
            call_func = getattr(requester_obj, func_name)
            res = None
            if call_func is not None:
                if args > 0:
                    res = call_func(*argv)
                else:
                    res = call_func()
            if res is not None:
                response_package = {"result": res, "src_rank": mpi_rank}
                # logger.info("sending message: {}".format(response_package))
                # msg = mpi_comm.isend(response_package, dest=dst, tag=response_tag)
                # msg.wait()
                mpi_comm.send(response_package, dest=msg_status.Get_source(), tag=response_tag)
    # logger.info("ABORTED ID Service")
