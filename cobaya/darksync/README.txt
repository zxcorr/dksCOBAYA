$ cd /data/AMARINS
COSMOLOGY: 
  (i) theoretical model
  - dksCOBAYA/cobaya/theories/darksync/darksync.py
INTERFACE DKS to COBAYA: 
  - dksCOBAYA/cobaya/theories/darksync/dkscobaya.py
dksCOBAYA launcher:
  (i)  Reads the YAML configuration file
  (ii) pass that configuration to Cobaya,
  (iii) start to run
  - dksCOBAYA/cobaya/darksync/scripts/dksCOBAYA_run.py

The additional structure is as follow:
         dksCOBAYA/
                cobaya/
                   theories/
                      darksync/
                         __init__.py
                         darksync.py
                         dkscobaya.py
                   darksync/
                      dksCOBAYA_run.py


After we have the three main files, create __init__ inside of the darksync folder:
(i) then python will treat it as a package and cobaya will load it
$ touch /data/AMARINS/COBAYA_modified/cobaya/cobaya/theories/darksync/__init__.py

You can also test:
$ cd /data/AMARINS/dksCOBAYA
$ python -c "from cobaya.theories.darksync.dkscobaya import dksLCDM; print(dksLCDM)"


After you have create/update the DKS files, and since we are in cobaya developing mode, you need to recompile it
$ cd /data/AMARINS/dksCOBAYA/cobaya/
$ pip install -e .


RUN:
$ cd /data/AMARINS/dksCOBAYA/cobaya/darksync
$ python scripts/dksCOBAYA_run.py yaml/dks_pantheon_test.yaml


cobaya-install cosmo -p /data/AMARINS/dksCOBAYA/cobaya_packages/




