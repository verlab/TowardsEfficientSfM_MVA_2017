#!/usr/bin/python

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import commands
import os
import subprocess
import sys

def check_dir(f):
    if not os.path.exists(f):
        os.makedirs(f)


verlabsfm = '/home/vnc/Dropbox/Projetos/Guilherme/Experiment_LBA_MVA/Verlab_SFM_v4/build/./VerLab_SFM '
#datasets = ['small_mine','small_city','intergeo'] 
#datasets = ['colombia_club','sand_mine','expopark']
datasets = ['small_mine','small_city','intergeo', 'colombia_club','sand_mine','expopark']
#datasets = ['castle_P19', 'castle_P30', 'entry-P10','fountain_P11', 'herz-jesu_P25', 'herz-jesu_P8']

experiment_name = "New_LBA"
full_exp = False

for dataset in datasets:

  print "\n --- running dataset ", dataset, " --- \n"

  dataset_path = "/local/Datasets/" + dataset
  #dataset_path = "/home/vnc/city_datasets/Stretcha_Dataset/" + dataset
  verlab_command = verlabsfm + dataset_path + "/sfm_params.txt "

  out_path = "/local/MVA_experiments/" + dataset + "/" + experiment_name

  check_dir(out_path)

  out_reg = out_path + "/out_reg.txt"
  out_recon = out_path + "/out_recon.txt"

  time_reg = out_path + "/time_reg.txt"
  time_recon = out_path + "/time_recon.txt"



  command_register_imgs = "{ time %s 1 |tee %s ;} 2> %s" % (verlab_command,out_reg,time_reg)
  command_recon = "{ time %s 2 |tee %s ;} 2> %s" % (verlab_command,out_recon,time_recon)

  if full_exp:
    proc = subprocess.Popen(['/bin/bash','-c',command_register_imgs])
    proc.wait()
    #print command_register_imgs

  proc = subprocess.Popen(['/bin/bash','-c',command_recon])
  proc.wait()
  #print command_recon


  #copy bundle.out and lists to directory

  command = "cp " + dataset_path + "/images/result/bundle.out " + out_path + " ; "
  command += "cp " + dataset_path + "/images/result/list.txt " + out_path

  proc = subprocess.Popen((str(command)), shell=True)
  proc.wait()
  #print command


sys.exit(1)

