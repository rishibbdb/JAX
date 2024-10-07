import multiprocessing
import subprocess


def start_process(argument_dict):
    script_path = '/home/storage/hans/jax_reco_new/extract_data_from_i3files/convert_i3_tfrecord_multiple_pulses_per_dom.py'
    cmd = ['python']
    cmd.append(f"{script_path}")

    for key in argument_dict.keys():
        val = argument_dict[key]
        cmd.append("--"+key)
        cmd.append(f"{val}")

    cmd.append("--recompute_true_muon_energy")
    return subprocess.run(cmd, shell=False)


if __name__ == '__main__':
    count = multiprocessing.cpu_count() // 2
    pool = multiprocessing.Pool(processes=count)

    arguments = []
    indir = f"/home/storage2/hans/i3files/21217/"
    outdir = f"/home/storage2/hans/i3files/21217/ftr/"
    infile_base = "wBDT_wDNN_L345_IC86-2016_NuMu"
    did = 21217

    delta=1000
    for i in range(35):
        min_file = i * delta
        max_file = (i+1) * delta

        argument_dict = dict()
        argument_dict['indir'] = indir
        argument_dict['infile_base'] = infile_base
        argument_dict['dataset_id'] = did
        argument_dict['outdir'] = outdir
        argument_dict['file_index_start'] = min_file
        argument_dict['file_index_end'] = max_file
        arguments.append(argument_dict)

    # start processes
    pool.map(start_process, arguments)
