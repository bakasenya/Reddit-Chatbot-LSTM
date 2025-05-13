import sqlite3
import pandas as pd

timeframes = ['2019-12']

for timeframe in timeframes:
    # Establish SQLite connection
    db_connection = sqlite3.connect(f'{timeframe}.db')
    cursor = db_connection.cursor()
    
    last_timestamp = 0  # Track the last 'unix' timestamp processed
    batch_size = 100000  # Number of rows to fetch per query
    current_batch_size = batch_size
    processed_count = 0
    is_test_done = False

    while current_batch_size == batch_size:
        # Fetch a batch of data
        query = f"SELECT * FROM parent_reply WHERE unix > {last_timestamp} AND parent IS NOT NULL AND score > 0 ORDER BY unix ASC LIMIT {batch_size}"
        data_frame = pd.read_sql(query, db_connection)

        # Update the last timestamp from the last row of the fetched batch
        last_timestamp = data_frame.tail(1)['unix'].values[0]
        current_batch_size = len(data_frame)

        # Write the data to appropriate files based on test completion status
        if not is_test_done:
            with open(f'{timeframe}.from', 'a', encoding='utf8') as from_file:
                for parent_text in data_frame['parent'].values:
                    from_file.write(parent_text + '\n')

            with open(f'{timeframe}.to', 'a', encoding='utf8') as to_file:
                for comment_text in data_frame['comment'].values:
                    to_file.write(str(comment_text) + '\n')

            is_test_done = True  # Mark test data as done

        else:
            with open(f'{timeframe}_train.from', 'a', encoding='utf8') as train_from_file:
                for parent_text in data_frame['parent'].values:
                    train_from_file.write(parent_text + '\n')

            with open(f'{timeframe}_train.to', 'a', encoding='utf8') as train_to_file:
                for comment_text in data_frame['comment'].values:
                    train_to_file.write(str(comment_text) + '\n')

        processed_count += 1
        if processed_count % 20 == 0:
            print(f'{processed_count * batch_size} rows processed so far')

    # Close the database connection
    db_connection.close()
