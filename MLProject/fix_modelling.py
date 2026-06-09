import re 
content = open('modelling.py', 'r', encoding='utf-8').read() 
content = re.sub(r'with mlflow.start_run\(run_name="RandomForest_CI"\):', 'with mlflow.start_run():', content) 
content = content.replace('    mlflow.set_experiment(EXPERIMENT_NAME)\n', '') 
open('modelling.py', 'w', encoding='utf-8').write(content) 
print('Berhasil!') 
