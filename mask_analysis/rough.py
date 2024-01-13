# %%
from pathlib import Path
import SimpleITK as sitk
from mask_analysis.utils import compress_fldr

import logging
from multiprocessing import Pool
from radiomics import collections
from fran.utils.fileio import load_dict
from fran.utils.helpers import get_pbar

from fran.utils.string import strip_extension

pbar=get_pbar()
def multiprocess_multiarg(func,arguments, num_processes=8,multiprocess=True,debug=False,progress_bar=True, logname = None):
    results=[]
    if multiprocess==False or debug==True:
        for res in pbar(arguments,total=len(arguments)):
            if debug==True:
                if logname:

                    logging.basicConfig(filename=logname,
                                        filemode='w',
                                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                        datefmt='%H:%M:%S',
                                        level=logging.DEBUG)
                    logging.info(" Processing {} .".format(res))
                # tr()
            results.append(func(*res,))
    else:
        p = Pool(num_processes)
        jobs = [p.apply_async(func=func, args=(*argument, )) for argument in arguments]
        p.close()
        pbar_fnc = get_pbar() if progress_bar==True else lambda x: x
        for job in pbar_fnc(jobs):
                results.append(job.get())
    return results

from random import random
from time import sleep
from multiprocessing import current_process
from multiprocessing import Process
import logging
 # SuperFastPython.com
# example of logging from multiple workers in the multiprocessing pool
from random import random
from time import sleep
from multiprocessing import current_process
from multiprocessing import Pool
from multiprocessing import Queue
from multiprocessing import Manager
from logging.handlers import QueueHandler
import logging
 
import pathlib
# executed in a process that performs logging
def logger_process(queue):
    # create a logger
    logfile_dir = pathlib.Path(__file__).parent
    logfile = logfile_dir.joinpath("applog.log")
    print("--------------------------------------------------------------------------\n\n\n")
    print(logfile)
    logger = logging.getLogger('app')
    # configure a stream handler
    logger.addHandler(logging.StreamHandler())
    logger.addHander(logging.FileHandler(filename=str(logfile), mode='w'))
    # log all messages, debug and up
    logger.setLevel(logging.DEBUG)
    # report that the logger process is running
    logger.info(f'Logger process running.')
    # run forever
    while True:
        # consume a log message, block until one arrives
        message = queue.get()
        # check for shutdown
        if message is None:
            logger.info(f'Logger process shutting down.')
            break
        # log the message
        logger.handle(message)
 
# task to be executed in child processes
def task(queue):
    # create a logger
    logger = logging.getLogger('app')
    # add a handler that uses the shared queue
    logger.addHandler(QueueHandler(queue))
    # log all messages, debug and up
    logger.setLevel(logging.DEBUG)
    # get the current process
    process = current_process()
    # report initial message
    logger.info(f'Child {process.name} starting.')
    # simulate doing work
    for i in range(5):
        # report a message
        logger.debug(f'Child {process.name} step {i}.')
        # block
        sleep(random())
    # report final message
    logger.info(f'Child {process.name} done.')
 
# protect the entry point
# %%
if __name__ == '__main__':
    # create the manager
    with Manager() as manager:
        # create the shared queue and get the proxy object
        queue = manager.Queue()
        # create a logger

        logger = logging.getLogger('app')
        # add a handler that uses the shared queue
        if not logger.hasHandlers():
            logger.addHandler(QueueHandler(queue))
        # log all messages, debug and up
        logger.setLevel(logging.DEBUG)
        # create the process pool with default configuration
        with Pool() as pool:
            # issue a long running task to receive logging messages
            _ = pool.apply_async(logger_process, args=(queue,))
            # report initial message
            logger.info('Main process started.')
            # issue task to the process pool
            results = [pool.apply_async(task, args=(queue,)) for i in range(5)]
            # wait for all issued tasks to complete
            logger.info('Main process waiting...')
            for result in results:
                result.wait()
            # report final message
            logger.info('Main process done.')
            # shutdown the long running logger task
            queue.put(None)
            # close the process pool
            pool.close()
            # wait for all tasks to complete (e.g. the logger to close)
            pool.join()
# task to be executed in child processes
# %%

# protect the entry point
 # %%

# %%
    fl1 = Path("/s/datasets_bkp/lits_segs_improved/images")
    fl2 = Path("/s/datasets_bkp/lits_segs_improved/masks")
    imgs = list(fl1.glob("*"))
    masks = list(fl2.glob("*"))
# %%
    fn = Path("/s/fran_storage/projects/litsmc/raw_dataset_properties.pkl")
    dici = load_dict(fn)
# %%
    compress_fldr(fl1)
# %%

    imgs = [strip_extension(fn.name) for fn in imgs]
    masks = [strip_extension(fn.name) for fn  in masks]

    my_list = masks
    duplicates = list(set([x for x in my_list if my_list.count(x)]))
    print(set(masks).difference(set(imgs)))
    print([item for item, count in collections.Counter(masks).items() if count > 1])
    len(set(masks))
# %%
    logger = logging.getLogger()
    # log all messages, debug and up
    logger.setLevel(logging.DEBUG)
    # report initial message
    logging.info('Main process started.')
    # configure child processes
    processes = [Process(target=task) for i in range(5)]
    # start child processes
    for process in processes:
        process.start()
    # wait for child processes to finish
    for process in processes:
        process.join()
    # report final message
    logging.info('Main process done.')

# %%
