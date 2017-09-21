# -*- coding: utf-8 -*-
import os
import click
import logging


def process_all_raw_folders(force, continue_failure):
    base_raw_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'data', 'raw')

    for raw_dir in next(os.walk(base_raw_dir))[1]:
        process_raw_folder(raw_dir, force, continue_failure)


def make_directory(directory_path, force):

    if os.path.exists(directory_path) and not os.path.isdir(directory_path):
        raise Exception("Cannot create output data folder because a file with the same name exists")

    if not force and os.path.isdir(directory_path) and not os.listdir(directory_path):
        raise Exception("Output folder " + proc_dir + " already exists.\nPass flag -f to force overwriting.")

    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)


def make_tfrecord(files_tuples, output_file, force=False):
    """Make a tfrecord and save it in output_file

    files_tuple: a list of tuples in the format (absolute_left_path, absolute_disp_path, absolute_conf_path)
    output_file: the tfrecords name
    force: overwrite a previous tfrecords file if exists

    """


def process_raw_folder(folder, force, continue_failure):
    """Implement the preprocessing data pipeline for a single raw folder.
    A raw folder contains three folders: left, disparity and gt.
    The three folders contain files with the same name.

    """
    logger = logging.getLogger(__name__)

    def maybe_stop(error, exception=None):
        if continue_on_failure:
            logger.warning(error)
        else:
            raise Exception(error)

    # Ensure that directories are correct

    raw_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'data', 'raw', folder)
    inter_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'data', 'interim', folder)
    proc_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'data', 'processed', folder)

    if not os.path.isdir(raw_dir):
        maybe_stop("Raw data folder " + raw_dir + " doesn't exist.")
        return

    try:
        make_directory(proc_dir, force)
    catch Exception as e:
        maybe_stop(str(e))
        return

    try:
        make_directory(inter_dir, force)
    catch Exception as e:
        maybe_stop(str(e))
        return

    # Create the confidence maps

    

    # Make csv file containing all files.
    # TODO handle error
    os.system( os.path.join(os.path.dirname(__file__), 'make_csv.sh') + " " + folder +
             " > " + os.path.join(inter_dir, 'data.csv') )




@click.command()
@click.argument('raw_subfolder', default=None)
@click.option('--force', '-f', is_flag=True, default=False)
@click.option('--continue-on-failure', '-i', is_flag=True, default=False)
def main(raw_subfolder, force, continue_on_failure):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')


    if raw_subfolder is None:
        logger.info('No subfolder was provided. Processing all raw data')
        process_all_raw_folders(force=force, continue_on_failure=continue_on_failure)
    else
        process_raw_folder(raw_subfolder, force=force, continue_on_failure=continue_on_failure)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)


    main()
