# generative-ai-applications

## Project description

[TO WRITE: in the agreed labelled-lines shape, one bold label per point and one sentence per line,
four to six lines, for example **The task:**, **The data:**, **The model:**, **The result:**.]

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
  notebook's Figure 1 gives the line for each photograph it shows.
- The Museum cannot warrant the quality or accuracy of the data, and this work has no endorsement,
  affiliation, support or approval from the Museum. The photographs are changed here: cropped to
  the specimen and scaled to 64 by 64 pixels.
- Selection: order Lepidoptera, project iCollections, the six butterfly families one query at a
  time, each record with one CC BY 4.0 specimen photograph, in a seed-2026 random order, taken until
  22,000 photographs had downloaded and verified.
- File: `data/manifest_fixed_20261013.csv`, one row per photograph, 22,000 rows, SHA-256
  `920fd9330771a12403a8cd07dea9b145dad633b7a86de81847b8a6c237cbbc80`. The photographs were
  downloaded on 13 October 2026, from 14:59 to 19:00 UTC. The manifest keeps no field that names a
  person.

## How to rebuild the data

1. The photographs. `scripts/download_nhm_images.py` runs once, outside the notebook, in its own
   environment with `requests` and Pillow (`pip install requests pillow`), because `requests` is
   not one of the notebook's tools. To rebuild the submitted set exactly, pass the fixed manifest as
   the candidates, with the target equal to its row count:

   ```bash
   python scripts/download_nhm_images.py download --candidates data/manifest_fixed_20261013.csv \
       --target 22000 --manifest data/manifest.csv
   python scripts/download_nhm_images.py verify
   ```

   The original download took about four hours at the script's polite pace of at most three
   workers, each pausing after every image, and a rebuild takes about as long.
2. The 64 by 64 arrays. If they are missing, the notebook's first Task 2 cell builds them from
   `data/raw/` with `scripts/build_working_set.py split`, which crops each photograph, flags likely
   crop failures and splits off the held-out tenth. That took 184 seconds on this rig with 12 worker
   processes. The notebook's call uses 8, so allow longer.

## What ships

- In this repository: the notebook under both names, the fixed manifest, both scripts,
  `requirements.txt` and this README.
- In the submission zip only: the three 64 by 64 arrays and their row table, `data/nhm_butterflies_64_training.npy`
  (187.5 MiB), `_held_out.npy` (20.7 MiB), `_flagged.npy` (49.6 MiB) and `_rows.csv` (2.3 MB),
  about 260 MB in all. They fit the 500 MB upload limit, but GitHub refuses files over 100 MiB.
- Not shipped: the 22,000 photographs, 7.4 GiB in `data/raw/`. The download script rebuilds them.

## How to run

These steps assume Python 3 and Git are installed. The notebook trains on a graphics processing
unit (GPU) when one is visible, and otherwise on the central processing unit (CPU).

1. Clone this repository and move into its folder.
2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   On Windows, activate it with `.venv\Scripts\activate` instead.

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Put the photographs in `data/raw/` (see above), because Figures 1 and 2 read ten of them. With
   the arrays from the submission zip in `data/` too, the notebook does not rebuild them.
5. Open the notebook, then restart the kernel and run all cells:

   ```bash
   jupyter notebook generative_model.ipynb
   ```

Run time on this rig, an NVIDIA GeForce RTX 5090: about 17 minutes of GPU training for the three
runs, and about 19 minutes for the whole notebook. On this rig's CPU the three training runs would
take between 4.5 and 10.9 hours. That is the notebook's estimate from timed training steps, not a
CPU run, and it moves mainly with the number of CPU threads: two runs on 16 threads gave 4.5 and
5.2 hours, and two on 24 threads gave 9.8 and 10.9 hours. The committed notebook shows 5.2 hours.

## How requirements.txt was created

The notebook was run top to bottom in a fresh virtual environment, with its packages (NumPy,
pandas, Matplotlib, Seaborn, PyTorch 2.14.1, Pillow and Jupyter) installed from the Python Package
Index, and the file was then written from that environment with:

```bash
pip freeze > requirements.txt
```

A second fresh environment, installed from `requirements.txt` alone, ran the notebook top to bottom
again.
