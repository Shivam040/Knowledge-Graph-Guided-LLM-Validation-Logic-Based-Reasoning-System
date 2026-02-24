## results.csv: Output of test.csv

## kdsh_trackA_data_centric Folder: Step by Step Data Processed with Advance summary
## Dossier: kdsh_trackA_data_centric\data\gold\run_id=run_20260112_111511\dossier
## src Folder: Code
## Logic/Pipeline: \src\kdsh\pipeline\steps


# (PoweShell Only) To run the file first create the docker image then run it in:

```
docker build --progress=plain -t kdsh:dev -f src/docker/Dockerfile .
```

### Option 1) time-taken(60 mins)

```
docker run --rm -it `
  -v "${PWD}:/workspace" `
  -v "${PWD}/src:/app/src" `
  -w /workspace `
  kdsh:dev `
  --train "/workspace/train.csv" `
  --test "/workspace/test.csv" `
  --novels "/workspace/novels/In search of the castaways.txt" `
           "/workspace/novels/The Count of Monte Cristo.txt" `
  --outdir "/workspace/kdsh_trackA_data_centric" `
  --run-all

```



### Option 2) time- taken(2 mins) please uses PowerShell line continuations (`) correctly

```
$env:RUN_ID = "run_001"

docker run --rm -it `
  --entrypoint python `
  -v "${PWD}:/workspace" `
  -w /workspace `
  -e PYTHONPATH=/workspace/src `
  -e RUN_ID=$env:RUN_ID `
  kdsh:dev `
  -c "from pathlib import Path; import pandas as pd;
from kdsh.pipeline.steps.step5_kg import step5_build_kg
from kdsh.pipeline.steps.step6_logic import step6_logic
from kdsh.pipeline.steps.step7_aggregate import step7_aggregate
from kdsh.pipeline.steps.step8_dossier import step8_dossier
from kdsh.pipeline.steps.step9_package import step9_package

run_id = __import__('os').environ.get('RUN_ID')
outdir = Path('/workspace/kdsh_trackA_data_centric')
silver = outdir/'data'/'silver'/f'run_id={run_id}'
gold = outdir/'data'/'gold'/f'run_id={run_id}'
gold.mkdir(parents=True, exist_ok=True)

chunks_df = pd.read_csv(silver/'chunks.csv')
claims_path = silver/'claims.jsonl'
facts_path = silver/'facts.jsonl'
retrieval_path = silver/'retrieval_candidates.csv'
evidence_path = silver/'evidence_labels.csv'
train_df = pd.read_csv('/workspace/train.csv')
test_df = pd.read_csv('/workspace/test.csv')

print('--- STEP5 ---')
b = set(p.name for p in silver.glob('*'))
step5_build_kg(chunks_df, claims_path, facts_path, silver, run_id, min_fact_conf=0.65)
a = set(p.name for p in silver.glob('*'))
print('new5:', sorted(a-b))

kg_path = next((silver/p for p in ['kg_triples.csv','kg_edges.csv'] if (silver/p).exists()), None)
print('kg_path:', kg_path)

print('--- STEP6 ---')
b = set(p.name for p in gold.glob('*'))
step6_logic(kg_path, train_df, test_df, gold, run_id)
a = set(p.name for p in gold.glob('*'))
new6 = sorted(a-b)
print('new6:', new6)

constraint_path = next((gold/p for p in new6 if 'constraint' in p.lower()), None)
if constraint_path is None:
    cand = sorted(gold.glob('*constraint*')) + sorted(gold.glob('*logic*'))
    constraint_path = cand[0] if cand else None
print('constraint_path:', constraint_path)

print('--- STEP7 ---')
decision_path, results_path, score = step7_aggregate(
    train_df, test_df, evidence_path, constraint_path,
    gold, run_id, contradiction_penalty=2.0
)
print('decision_path:', decision_path)
print('results_path:', results_path)
print('score:', score)

print('--- STEP8 ---')
dossier_path, dossier_json = step8_dossier(
    chunks_df, claims_path, retrieval_path, evidence_path,
    decision_path, gold, run_id,
    key_claims_per_id=4, evidence_rows_per_claim=3
)
print('dossier_path:', dossier_path)
print('dossier_json:', dossier_json)

print('--- STEP9 ---')
manifest_path = outdir/'data'/'run_manifests'/f'{run_id}.json'
zip_path = step9_package(gold, manifest_path, run_id)
print('zip_path:', zip_path)
print('DONE')"
```# Knowledge-Graph-Guided-LLM-Validation-Logic-Based-Reasoning-System
