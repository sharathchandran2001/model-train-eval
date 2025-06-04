from evidently.test_suite import TestSuite
from evidently.tests import *

def evaluate_data_quality(train_df, test_df):
    suite = TestSuite(tests=[
        TestNumberOfMissingValues(),
        TestColumnDrift(column_name=train_df.columns[0])
    ])
    suite.run(train_data=train_df, current_data=test_df)
    return suite.as_dict()
