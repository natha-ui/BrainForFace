# One neuron, one face: face cells in a simulated CL1 culture

Can a single neuron in a small-world leaky integrate-and-fire (LIF) culture learn,
without supervision, to fire for one person's face and nobody else's? The test is
modelled on human "concept cells" (Quiroga et al. 2005).

Developed during Outlier Labs Biocomputing Hackathon using simulated Cortical Labs CL1 

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


## Files
`face_cells/encode.py` images and encoders · `culture.py` LIF wrapper, STDP, inhibition, safety ·
`analysis.py` statistics · `run.py` protocol · `figures.py` diagnostic figures ·
`visualise.py` poster · `query.py` photo → neuron · `crop_faces.py` · `make_placeholders.py`.
