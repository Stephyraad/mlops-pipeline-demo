import os

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

def build_drift_report(
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        output_path: str
    ) -> None:

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference_data, current_data=current_data)
    snapshot.save_html(output_path)


def main() -> None:
    print()

if __name__ == "__main__":
    main()