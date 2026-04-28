"""Merge v_ion arrays into the existing gpaw_qm9 HDF5 alongside n_r."""
import h5py, sys

def merge(vion_path, main_path):
    with h5py.File(vion_path, "r") as vion_db, \
         h5py.File(main_path, "a") as main_db:
        for key in vion_db:
            if key in main_db:
                grp = main_db[key]
                if "v_ion" in grp:
                    del grp["v_ion"]
                data = vion_db[key]["v_ion"][:]
                
                if "n_r" in grp:
                    n_r_shape = grp["n_r"].shape
                    if data.shape != n_r_shape:
                        print(f"WARN {key}: v_ion {data.shape} != n_r {n_r_shape}, skipping")
                        continue
                
                grp.create_dataset("v_ion", data=data,
                                   compression="gzip", compression_opts=4)
        print("Done merging v_ion")

if __name__ == "__main__":
    merge(sys.argv[1], sys.argv[2])
