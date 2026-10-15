# generative-ai-applications

## Project description

**The task:** I trained a deep convolutional generative adversarial network (DCGAN) from scratch to generate new 64 by 64 colour images of pinned butterflies.

**The data:** I used 22,000 Natural History Museum photographs of British and Irish butterfly specimens, licensed CC BY 4.0, and 19,800 of them trained the model.

**The model:** The generator and discriminator follow my coursework DCGAN, and one training run took about six minutes on one graphics card.

**The result:** Most of the 64 numbered samples read as plausible museum specimens, and none crossed my nearest-neighbour copy threshold.

**The real-world test:** I tried the trained discriminator as a quality flag for museum digitisation and it did not work, so my next step would be a dedicated anomaly model trained only on normal photographs.

## The notebook and the report

- `generative_model.ipynb` and `generative_ai.ipynb` are the same notebook, byte for byte, under the
  two names the brief uses.
- The report ships as `Generative_AI_Analysis_Report.pdf` and, with the same content, as
  `module_summary.pdf`, the two names the brief uses.
- No Fréchet Inception Distance is computed, because it needs a network pre-trained on ImageNet and
  this project trains every network itself. The qualitative evaluation is the core, and the two
  simple measures in Task 4 are labelled supplementary.
- The rubric's Written Report Quality row is cut off at "It communicates tec", so its full wording
  was not available. I used its likely meaning from the matching rows in earlier projects of the
  brief, that the report should be clear to both technical and non-technical readers, and I wrote
  it for both.

## Dataset

- Name: Natural History Museum, Collection specimens, the iCollections British and Irish
  butterflies (Paterson et al., 2016): specimen photographs of pinned butterflies.
- Source: the Natural History Museum Data Portal, https://data.nhm.ac.uk/dataset/collection-specimens,
  Specimens resource `05ff2255-c38a-40c9-b657-4ccb55ab2feb`.
- Licence: the photographs are under Creative Commons Attribution 4.0 (CC BY 4.0,
  https://creativecommons.org/licenses/by/4.0), checked on each image's own licence field, and the
  records are under Creative Commons Zero (CC0).
- Attribution, in the form the Museum asks for: "Natural History Museum. (Year updated). Image
  title (from Collection specimens) [Photograph]. Natural History Museum. Licensed under CC-BY-4.0
  (https://creativecommons.org/licenses/by/4.0). https://data.nhm.ac.uk/object/occurrence-id". The
  notebook gives the line under Figures 1 and 2 for each photograph they show, and
  `data/figure_photos/ATTRIBUTION.md` gives it for each of the ten photographs shipped here.
- The Museum cannot warrant the quality or accuracy of the data, and this work has no endorsement,
  affiliation, support or approval from the Museum. For training, the photographs are changed:
  cropped to the specimen and scaled to 64 by 64 pixels. The ten in `data/figure_photos/` are
  changed too: labels masked.
- Selection: order Lepidoptera, project iCollections, the six butterfly families one query at a
  time, each record with one CC BY 4.0 specimen photograph, in a seed-2026 random order, taken until
  22,000 photographs had downloaded and verified.
- File: `data/manifest_fixed_20261013.csv`, one row per photograph, 22,000 rows, Secure Hash
  Algorithm 256 (SHA-256) `920fd9330771a12403a8cd07dea9b145dad633b7a86de81847b8a6c237cbbc80`. The
  photographs were downloaded on 13 October 2026, from 14:59 to 19:00 Coordinated Universal Time
  (UTC). The manifest keeps no field that names a person.

## How to rebuild the data

1. The photographs. `scripts/download_nhm_images.py` runs once, outside the notebook, in its own
   environment with `requests` and Pillow (`pip install requests pillow`), because `requests` is
   not one of the notebook's tools. To rebuild the submitted set exactly, check the fixed
   manifest's hash, pass the manifest as the candidates with the target equal to its row count,
   then check every photograph against the hash its row gives:

   ```bash
   sha256sum data/manifest_fixed_20261013.csv
   python scripts/download_nhm_images.py download --candidates data/manifest_fixed_20261013.csv \
       --target 22000 --manifest data/manifest.csv --no-fallback
   python scripts/download_nhm_images.py verify --manifest data/manifest_fixed_20261013.csv
   ```

   The first command should print the SHA-256 given above. `--no-fallback` fetches each photograph
   only from the address its row gives, so the bytes match. The last command re-hashes every
   photograph in `data/raw/` against the fixed manifest and checks each one's licence. The original
   download took about four hours at the script's polite pace of at most three workers, each
   pausing after every image, and a rebuild takes about as long.
2. The 64 by 64 arrays. If they are missing, the notebook's first Task 2 cell builds them from
   `data/raw/` with `scripts/build_working_set.py split`, which crops each photograph, flags likely
   crop failures and splits off the held-out tenth. That took 184 seconds on this rig with 12 worker
   processes. The notebook's call uses 8, so allow longer. The notebook then checks the manifest,
   the arrays and the row table against the hashes frozen for the submitted run, and stops if one
   differs. Its `REPRODUCE` setting is `True` for that; setting it to `False` trains on a new set
   of photographs and only warns.

## What ships

Sizes are given in mebibytes (MiB), gibibytes (GiB) and megabytes (MB).

- In this repository: the notebook under both names, the report under both names
  (`Generative_AI_Analysis_Report.pdf` and `module_summary.pdf`), the fixed manifest, both scripts,
  `requirements.txt` and this README. It also holds the ten photographs Figures 1 and 2 read, in
  `data/figure_photos/` (2.2 MiB), with `ATTRIBUTION.md` giving the Museum's CC BY 4.0
  attribution line for each. They are modified: labels masked. `scripts/build_working_set.py mask`
  painted each one's label section grey, so no collector's name or handwriting is published, and
  `masks.csv` there gives the box and side each was masked by. The download script rebuilds the
  originals in `data/raw/`.
- Also in this repository: `data/crop_audit_600.csv`, the crop audit's verdict for each of the 600
  crops scored by eye before training. Each row gives the verdict (pass, doubtful or fail), the kind
  of failure and who scored it. An AI assistant scored the crops, and I reviewed the results. Task 2
  counts the failure rate and the crop flag's agreement from this file. The repository also holds
  `data/candidates_skipped.csv`, which gives the 2,635 records left out when the candidates were
  listed, each with its occurrence ID, catalogue number and reason. Task 2 counts them.
- The submitted checkpoints, also in this repository: `models/generator_ema.pt` (13.7 MiB), run
  1's exponential moving average (EMA) generator at epoch 100, and `models/discriminator.pt`
  (10.6 MiB), its discriminator.
  Task 4 saves them there and loads them from there. The model the checkpoint rule picked as
  written, run 3 at epoch 5, stays local in `models/rule_as_written/`, with every run's
  checkpoints in `outputs/`.
- In the submission zip only: the three 64 by 64 arrays and their row table,
  `data/nhm_butterflies_64_training.npy` (187.5 MiB), `_held_out.npy` (20.7 MiB), `_flagged.npy`
  (49.6 MiB) and `_rows.csv` (2.3 MB), about 273 MB, or 260 MiB, in all. They fit the 500 MB upload
  limit, but GitHub refuses files over 100 MiB.
- Not shipped: the other 21,990 photographs, 7.3 GiB in `data/raw/`. The download script rebuilds
  them.

## How to run

These steps assume Linux, or Windows through Windows Subsystem for Linux (WSL2), with Python 3.12
(tested on 3.12.3), an NVIDIA graphics processing unit (GPU) with CUDA, and Git. That is the
environment `requirements.txt` was frozen from, so it pins Linux CUDA wheels, such as `triton`
and the `nvidia-*` packages, which do not install on native Windows or macOS. The PyTorch wheels
are built for CUDA 13.0, so the GPU's driver must support CUDA 13.0, or PyTorch sees no GPU. This
rig's driver supports CUDA 13.4. Without a GPU the notebook trains on the central processing unit
(CPU), far more slowly.

This rig is an NVIDIA GeForce RTX 5090 with 32,607 MiB of memory, an Intel Core Ultra 9 285K
processor with 24 cores, and 93.9 GiB of random access memory (RAM) as WSL sees it. The notebook
counts a machine as this rig only when the GPU's full name, its memory and the RAM all match. On
any other machine, the training watchdog only warns on a slow epoch, high memory or the 30-minute
budget, and never stops a run for them.

1. Clone this repository and move into its folder:

   ```bash
   git clone https://github.com/PaddyGilliland1/generative-ai-applications-project
   cd generative-ai-applications-project
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   On Windows, run every step inside WSL2.

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Put the three arrays and the row table from the submission zip in `data/`. Figures 1 and 2 read
   their ten photographs from `data/figure_photos/`, which ships here. The notebook needs the full
   set in `data/raw/` (see above) only if the arrays are missing, to rebuild them.
5. Open the notebook, then restart the kernel and run all cells:

   ```bash
   jupyter notebook generative_model.ipynb
   ```

Run time on this rig, an NVIDIA GeForce RTX 5090: about 22 minutes of GPU training for the three
runs, and about 23 minutes for the whole notebook. On this rig's CPU the three training runs would
take about 4.8 hours, from 0.38 seconds a training step on 16 threads. That is the notebook's
estimate from timed training steps, not a CPU run, and it moves with the number of CPU threads and
how busy the machine is.

On other hardware, or on the CPU, the same seeds can still train a different model and draw
different samples. The scoring cell in Task 4 checks the samples' pixels against the hash of the
submitted samples, and if they differ it warns and hides the scores and their reasons, which
describe the submitted samples only. The submitted samples, and every result the report quotes,
are the outputs saved in the committed notebook.

## How requirements.txt was created

The notebook was run top to bottom in a fresh virtual environment, with its packages (NumPy,
pandas, Matplotlib, Seaborn, PyTorch 2.14.1, Pillow and Jupyter) installed from the Python Package
Index, and the file was then written from that environment with:

```bash
pip freeze > requirements.txt
```
