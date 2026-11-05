# One neuron, one face: face cells in a simulated CL1 culture

Can a single neuron in a small-world leaky integrate-and-fire (LIF) culture learn,
without supervision, to fire for one person's face and nobody else's? The test is
modelled on human "concept cells" (Quiroga et al. 2005). See `PROMPT.md` for the
original prompts and the decisions made along the way.

![poster](results/face_cells_poster.png)

> Simulator results are not evidence about CL1 tissue.

## Run it in VS Code

Requires **Python 3.12 or 3.13**.

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate   |   Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py            # ~4-5 min on a laptop; writes results/
```

Or open the folder in VS Code, choose the `.venv` interpreter (Ctrl/Cmd+Shift+P,
"Python: Select Interpreter"), then press **F5** and pick a launch configuration:

| Launch config | What it does |
|---|---|
| **Full pipeline** | experiment → figures → poster → example queries (`run_all.py`) |
| **Quick smoke test** | same, but ~1 min (fewer epochs and views, no no-inhibition control) |
| **Query a photo** | asks the trained culture which face cell fires for a photo you choose |
| **Real-time demo (CL SDK + live viewer)** | presents held-out faces through `cl.open()` / `neurons.loop()` at wall-clock speed; the SDK's web viewer shows live spikes and stims on the 8x8 array |
| **Real-time: identify a photo** | same, for a photo you choose |
| **Poster only** | re-renders `results/face_cells_poster.png` from existing results |
| **Crop raw photos** | `data/raw/<person>/*.jpg` → `data/faces/` |

There is also a **Terminal → Run Task → Install requirements** task.

### Your own faces
Put photos in `data/raw/<person>/` (any number of people; ~10+ photos each, mixed
poses), then run `python run_all.py --raw data/raw`. Every third photo of each
person is held out and never trained on, so a face cell has to generalise to new poses.

## Real-time mode (CL SDK, as in natha-ui/biocompute-outlier-labs)

`face_cells/realtime.py` runs the trained culture through the Cortical Labs SDK loop,
the same path as that repo's `experiments/example1.py`. The SDK spawns the LIF simulator
as a subprocess and runs it in real time, with the live visualiser at
`http://127.0.0.1:8000/visualiser`. Each face is sent as CL1-valid commands: one
`StimDesign(200, -a, 200, a)` per input electrode (0.1 µA steps), delivered as
`BurstDesign(10, 200 Hz)`. The SDK subprocess doesn't expose `apply_reward`, so learning
happens offline in `run.py`; real-time mode loads the learned weights (plus inhibition)
as the simulator's `connectivity`. On a CL1, drop the `set_simulator_data_source` call
and the same loop would stimulate tissue.

```bash
python face_cells/realtime.py                       # live demo, opens the viewer
python face_cells/realtime.py --image my_photo.jpg  # who is this?
CL_SDK_ACCELERATED_TIME=1 python face_cells/realtime.py --no-browser   # fast, no viewer
```

## What happens

1. **Encode.** Crop the face, downsample to 8×4, subtract the average face and keep
   the brighter-than-average parts (norm-based coding). Pixel brightness sets the
   amplitude of 10 biphasic pulses at 200 Hz (the CL1 burst limit; ≤1.5 µA) on input electrodes in
   columns 0–3. The output neurons in columns 4–7 are never stimulated.
2. **Culture.** The vendored `organoid_simulator` from wbmills/biocompute-outlier-labs
   (MIT): 64 LIF neurons, small-world coupling at 7 mV, background noise 0.5 mV.
   Added here: trace-STDP with a constant reward (unsupervised), synaptic scaling,
   and lateral inhibition between output cells (4 mV). A safety guard skips learning
   on burst trials and backs off stimulation.
3. **Protocol.** Blank baseline → pre-probe → STDP exposure (15 epochs) → blank
   baseline → post-probe. Order is counterbalanced. Controls: the same culture with
   plasticity frozen, and STDP without inhibition.
4. **Test.** A face cell responds above baseline + 5 SD to one person only, on
   held-out photos, and is chosen on training photos. The report also gives z, SI,
   sparseness, AUC and a label-shuffle permutation test.

## Reference result (`results/`, seed 42, 200 Hz bursts)

Data: 124 photos of 3 people (personA 51, including hair down and glasses; personB 38;
personC 35). Every third photo is held out. Counterbalanced order, 4 mV inhibition.

* **STDP recruits ch34 as personA's cell:** z 8.1, AUC 0.73, permutation p 0.004.
  The frozen control falls back on hub ch35, which fires for everyone. This is the most
  repeatable finding: ch34 appeared in every run (both seeds, 200 and 250 Hz). It passed
  the strict test (this person only, >5 SD, held-out poses incl. glasses) once, at 250 Hz
  with seed 42 (z 7.8). Here it misses by 0.02 spikes.
* **personC:** best cell ch36 (z 6.2) is not selective. **personB:** no cell (z ≤ 1).
* **Real-time SDK check:** ch34 fires for personA (z 5.8) and stays below threshold for
  others (personC 4.8), but personC's ch36 fires for every face, so verdicts are mixed.
  Offline and real-time agree.
* **Network stability:** 0 disrupted trials in every inhibited run.
* **Caveats:** a few seeds and 3 people. Single-cell selectivity is fragile and can flip
  with seed or presentation order. Treat this as a mechanism demo.

## Files
`face_cells/encode.py` images and encoders · `culture.py` LIF wrapper, STDP, inhibition, safety ·
`analysis.py` statistics · `run.py` protocol · `figures.py` diagnostic figures ·
`visualise.py` poster · `query.py` photo → neuron · `crop_faces.py` · `make_placeholders.py`.
