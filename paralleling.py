import multiprocessing, time

def do_paralleling(paths, do_func):
    # Create a pool of worker processes
    pool = multiprocessing.Pool()

    # Process the images in parallel
    start_time = time.time()
    processed = pool.map(do_func, paths)
    elapsed_time = time.time() - start_time

    # Close the pool of worker processes
    pool.close()
    pool.join()

    return processed, elapsed_time