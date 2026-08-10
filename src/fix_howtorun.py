with open('README.md', 'r', encoding='utf-8') as f:
    content = f.read()

old_section = """## How to run

pip install -r requirements.txt
python src/clean_data.py
python src/feature_engineering.py
python src/encode_features.py
python src/scale_features.py
python src/train_logistic.py
python src/threshold_tuning.py
python src/train_lightgbm.py
python src/threshold_tuning_lgbm.py
python src/save_logistic.py"""

new_section = """## How to run

    pip install -r requirements.txt
    python src/clean_data.py
    python src/feature_engineering.py
    python src/encode_features.py
    python src/scale_features.py
    python src/train_logistic.py
    python src/threshold_tuning.py
    python src/train_lightgbm.py
    python src/threshold_tuning_lgbm.py
    python src/save_logistic.py"""

if old_section in content:
    content = content.replace(old_section, new_section)
    with open('README.md', 'w', encoding='utf-8', newline='\n') as f:
        f.write(content)
    print("Fixed: How to run section now uses proper code formatting")
else:
    print("Could not find the exact section to replace - no changes made")