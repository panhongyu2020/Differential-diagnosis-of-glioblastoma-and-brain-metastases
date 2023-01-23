import pandas as pd

def classifaction_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-4]:
        if len(line.strip()) == 0:
            continue
        row = {}
        row_data = line.split('      ')
        if float(row_data[5].strip()) == 0.0:
            continue
        row['class'] = row_data[1].strip()
        row['precision'] = float(row_data[2].strip())
        row['recall'] = float(row_data[3].strip())
        row['f1_score'] = float(row_data[4].strip())
        row['support'] = float(row_data[5].strip())
        report_data.append(row)
    df = pd.DataFrame.from_dict(report_data)
    return df

def parse_acc_from_classifaction_report(report):
    lines = report.split('\n')
    row_data = lines[-3].split('      ')
    precision = float(row_data[1])
    f1_score = float(row_data[3])
    return f1_score, precision



