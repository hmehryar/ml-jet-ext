#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2024–2025
#   Authors: Haydar Mehryar & Ritoban Datta
#   Wayne State University — HPC Lab
import os
import xml.etree.ElementTree as ET
# import get_pthat_binsOriginal as gpt
def load_problem_ids(problems_csv):
    if not problems_csv:
        return None
    import re
    try:
        with open(problems_csv, "r") as f:
            txt = f.read().strip()
        ids = [int(t) for t in re.split(r"[,\s]+", txt) if t]
        return ids if ids else None
    except Exception as e:
        print(f"[WARN] Could not read '{problems_csv}': {e}")
        return None
def main(pthat_index, FolderName, problems_csv=None):
    # XML file path
    os.makedirs(f"{FolderName}", exist_ok=True)
    # pthat_bins = gpt.GetPtHatBins(5020)
    xml_file = f"{FolderName}/jet_main.xml"
    tree = ET.parse(xml_file)
    root = tree.getroot()

    problem_file_path = os.path.join(FolderName, problems_csv) if problems_csv else None
    problem_ids = load_problem_ids(problem_file_path)
    # Iterate over events
    # for k in range(0,200):
    print(f"[INFO] Using problem IDs: {problem_ids}")
    


    for k in (problem_ids if problem_ids is not None else range(0, 200)):
        #if k in [0,1,4,5,6,7]:
        #    continue
        xml_file = f"{FolderName}/jet_main.xml"
        tree = ET.parse(xml_file)
        root = tree.getroot()
        new_event_number = k
        for element in root.findall(".//outputFilename"):
            current_text = element.text.strip()
            print(current_text)
            if "event" in current_text:
                # Update outputFilename with event number
                base_path, _ = current_text.rsplit("event", 1)
                #element.text = f"{base_path}/{pthat_bins[pthat_index][0]}_{pthat_bins[pthat_index][1]}_{new_event_number}"
                element.text = f"{base_path}/jetmod_{new_event_number}"
                #element.text = f"/wsu/home/hl/hl97/hl9735/pp_resultsXscape/pp_{new_event_number}"
                base_folder = os.path.dirname(base_path)
                #new_xml_file = os.path.join(base_folder, f"/wsu/tmp/hl9735/{FolderName}/{pthat_bins[pthat_index][0]}_{pthat_bins[pthat_index][1]}_Run{k}.xml")
                new_xml_file = f"{FolderName}/jetmod_Run{k}.xml"
                #new_xml_file = f"/wsu/tmp/hl9735/{FolderName}/{pthat_bins[pthat_index][0]}_{pthat_bins[pthat_index][1]}_Run{k}.xml"
                # Update the seed value under <Random>
                random_element = root.find(".//Random/seed")
                if random_element is not None:
                    random_element.text = str(new_event_number)  # Set seed to match new_event_number
                    print(f"Updated seed to {new_event_number}.")

                #hydro_element = root.find(".//Hydro/Brick")
                #print(ET.tostring(hydro_element))  # See if it's finding the element
                #if hydro_element is not None:
                #    temp = hydro_element.find("T")
                #    if temp is not None:
                #        temp.text = str(0.15 + 0.05 * (k // 5.0))
                #matter_element = root.find("./Eloss/Matter")
                #if matter_element is not None:
                #   tempup = matter_element.find("TempUp")
                #    tempdown = matter_element.find("TempDown")
                #    if tempup is not None and tempdown is not None:
                #        tempup.text = str(0.15 + 0.05 * (k // 5.0))
                #        tempdown.text = str(0.15 + 0.05 * (k // 5.0))
              #  print(temp.text, tempup.text, tempdown.text)
              #  pythia_element = root.find(".//Hard/PythiaGun")
              #  if pythia_element is not None:
              #      pthat_min_element = pythia_element.find("pTHatMin")
              #      pthat_max_element = pythia_element.find("pTHatMax")
              #      if pthat_min_element is not None and pthat_max_element is not None:
                #        index = k // 7
              #          pthat_min_element.text = str(pthat_bins[pthat_index][0])
              #          pthat_max_element.text = str(pthat_bins[pthat_index][1])

                # Save the updated XML file
                tree.write(new_xml_file, encoding="utf-8", xml_declaration=True)
                print(f"Updated XML has been saved to '{new_xml_file}'.")
                break

        # Define job-specific paths and parameters
        #build_dir = f"/wsu/tmp/hl9735/{FolderName}/build_{pthat_bins[pthat_index][0]}_{pthat_bins[pthat_index][1]}_{k}"
        build_dir = f"{FolderName}/build_{k}"
        #jetscape_xml = f"/mnt/tmp/{pthat_bins[pthat_index][0]}_{pthat_bins[pthat_index][1]}_Run{k}.xml"
        jetscape_xml = f"/mnt/tmp/jetmod_Run{k}.xml"
        #bash_script_path = f"/wsu/tmp/hl9735/{FolderName}/job_script_{pthat_bins[pthat_index][0]}_{pthat_bins[pthat_index][1]}_{k}.sh"
        bash_script_path = f"{FolderName}/job_script_{k}.sh"

        # Create the bash script for the job
        bash_script = f"""#!/bin/bash
set -ex
module load apptainer
export JETSCAPE_DIR=/wsu/home/gy/gy40/gy4065/ml-jet-ext/JETSCAPE/
#export APPTAINERENV_LHAPATH=/wsu/home/hl/hl97/hl9735/X-SCAPE/external_packages/LHAPDF
#export SMASH_DIR=${{JETSCAPE_DIR}}/external_packages/smash/smash_code

# Create build directory
mkdir -p {build_dir}
cd {build_dir}
#rm CMakeCache.txt
# Configure and build
apptainer exec --cleanenv \\
  --bind /rs/rs_grp_majshen/tier1/JETSCAPEDataFile/HydroProfiles/5020GeV_0-5/:/mnt/home \\
  --bind {FolderName}:/mnt/tmp \\
  --bind "{FolderName}:{FolderName}" \\
  --bind "${{JETSCAPE_DIR}}:${{JETSCAPE_DIR}}" \\
  /wsu/home/gy/gy40/gy4065/base_stable.sif \\
  cmake $JETSCAPE_DIR 

apptainer exec --cleanenv \\
  --bind /rs/rs_grp_majshen/tier1/JETSCAPEDataFile/HydroProfiles/5020GeV_0-5/:/mnt/home \\
  --bind {FolderName}:/mnt/tmp \\
  --bind "{FolderName}:{FolderName}" \\
  --bind "${{JETSCAPE_DIR}}:${{JETSCAPE_DIR}}" \\
  /wsu/home/gy/gy40/gy4065/base_stable.sif \\
  make -j4

# Wait until runJetscape exists (max 5 attempts with 2s interval)
for i in {{1..5}}; do
  if [ -f runJetscape ]; then
    break
  else
    sleep 2
  fi
done

if [ ! -f runJetscape ]; then
  echo "Error: runJetscape not found after build."
  exit 1
fi

# Run simulation
apptainer exec --cleanenv \\
  --bind /rs/rs_grp_majshen/tier1/JETSCAPEDataFile/HydroProfiles/5020GeV_0-5/:/mnt/home \\
  --bind {FolderName}:/mnt/tmp \\
  --bind "{FolderName}:{FolderName}" \\
  --bind "${{JETSCAPE_DIR}}:${{JETSCAPE_DIR}}" \\
  /wsu/home/gy/gy40/gy4065/base_stable.sif \\
  ./runJetscape {jetscape_xml} /wsu/home/gy/gy40/gy4065/ml-jet-ext/JETSCAPE/config/jetscape_main.xml

# Cleanup
rm -r {build_dir}
"""

        # Write the bash script to a file
        with open(bash_script_path, "w") as f:
            f.write(bash_script)

        os.chmod(bash_script_path, 0o755)  # Make it executable
        
        # Define SLURM job parameters
        #job_name = f"JETSCAPE_{k}_{pthat_bins[pthat_index][0])}_{pthat_bins[pthat_index][1]}"
        #job_name = f"JSM0010_{pthat_bins[pthat_index][0]}_{pthat_bins[pthat_index][1]}_{k}"
        # job_name = f"c0005_{k}"

        # Extract alpha_s and q_0 values from folder name (e.g. c0005_alpha0p20_q_1p5)
        alpha_str = FolderName.split("alpha")[-1].split("_q_")[0]
        q0_str = FolderName.split("_q_")[-1]
        job_name = f"c0005_a{alpha_str}_q{q0_str}_{k}"


        out_log = f"{FolderName}/log/{job_name}.out"
        err_log = f"{FolderName}/log/{job_name}.err"
        sbatch_command = (
            f"sbatch -q primary -N 1 -n 1 --mem=4G -t 30:00:00 "
            f"--job-name={job_name} -o {out_log} -e {err_log} {bash_script_path}"
        )

        # Submit the jo
        print(f"Submitting job: {job_name}")
        result = os.system(sbatch_command)

        # Check submission result
        if result == 0:
            print(f"Job {job_name} submitted successfully.")
        else:
            print(f"Error submitting job {job_name}. Exit status: {result}")
        
        
    return 0
#To delete the content of the directory and rerunning the script, use the following command
#rm -rf log/ build_*/ jetmod_Run*.xml job_script_*.sh  jetmod_*_final_state_hadrons.dat

#To run the script, use the following command
# main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p20_q_1p0") #Backlog | InProgress | Done
# main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p20_q_1p0", "problem_jobs.csv") #Backlog | InProgress | Done

#main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat100-110_eCM5020/c0005_alpha0p20_q_1p5") #Backlog | InProgress | Done
#main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat100-110_eCM5020/c0005_alpha0p20_q_1p5", "problem_jobs.csv") #Backlog | InProgress | Done


#The main function can be in the Backlog or InProgress or Done status depending on how many jobs have been submitted and completed.

if __name__ == "__main__":
    for i in range(42,43):

        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p20_q_1p0", "problem_jobs.csv") #InProgress | Done
        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p20_q_1p5", "problem_jobs.csv") #InProgress | Done
        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p20_q_2p0", "problem_jobs.csv") #InProgress | Done
        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p20_q_2p5", "problem_jobs.csv") #InProgress | Done
        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p20_q_3p0", "problem_jobs.csv") #InProgress | Done

        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p25_q_1p0") #InProgress
        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p25_q_1p5") #InProgress
        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p25_q_2p0") #Backlog
        main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p25_q_2p5") #Backlog
        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p25_q_3p0") #Backlog

        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p30_q_1p0") #Backlog
        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p30_q_1p5") #Backlog
        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p30_q_2p0") #Backlog
        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p30_q_2p5") #Backlog
        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p30_q_3p0") #Backlog

        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p35_q_1p0") #Backlog
        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p35_q_1p5") #Backlog
        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p35_q_2p0") #Backlog
        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p35_q_2p5") #Backlog
        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p35_q_3p0") #Backlog

        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p40_q_1p0") #Backlog
        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p40_q_1p5") #Backlog
        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p40_q_2p0") #Backlog
        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p40_q_2p5") #Backlog
        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p40_q_3p0") #Backlog

        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p45_q_1p0") #Backlog
        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p45_q_1p5") #Backlog
        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p45_q_2p0") #Backlog
        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p45_q_2p5") #Backlog
        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p45_q_3p0") #Backlog

        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p50_q_1p0") #Backlog
        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p50_q_1p5") #Backlog
        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p50_q_2p0") #Backlog
        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p50_q_2p5") #Backlog
        # main(i, "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p50_q_3p0") #Backlog
