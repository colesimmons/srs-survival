"""
- Take cleaned up review history
- Add time-related features (time since prev review + time since first review)
- Normalize time-related features
- Drop cards with no rows where no review_type = Review (i.e never learned it)
- Drop 'timestamp'
- Write to CSV

Input columns:
- card_id | timestamp | was_remembered | answer_score | secs_to_answer | review_type (| front) (| back)

Add columns:
- time_since_first_review {float} -- time since first review of flashcard
    normalized to mean 0, std 1 using StandardScaler
- time_since_prev_review_minmax {float} -- time since previous review of flashcard
    normalized to [0, 1] using MinMaxScaler
    used for targets because times provided to Weibull survival model must be positive
- time_since_prev_review_standard {float} -- time since previous review of flashcard
    normalized to mean 0, std 1 using StandardScaler
    used for input sequences
- time_to_answer {float} -- just secs_to_answer but normalized to mean 0, std 1 using StandardScaler
    renamed because it no longer represents seconds

Drop columns:
- timestamp
- secs_to_answer 
"""


import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


output_file = 'review_history_with_time_features.csv'


def main(input_filename = "review_history.csv"):
  # Load CSV of individual reviews
  df = pd.read_csv(input_filename, parse_dates=['timestamp'])

  # Sort reviews by card ID -> timestamp in case they weren't already
  df.sort_values(['card_id', 'timestamp'], ascending=True, inplace=True)

  # Drop cards with no rows where no Review Type = Review (i.e never learned it)
  df = df.groupby('card_id').filter(lambda x: (x['review_type'] == 'Review').any())

  # Add column: time since prev card review (0 if first card review)
  # If you remove scaling, you might want to divide by 60, 3600, etc. to get minutes, hours, etc.
  # But as long as you're scaling, it doesn't matter
  df['time_since_prev_review'] = df.groupby('card_id')['timestamp'].diff().dt.total_seconds().fillna(0)
  
  # Add column: time since first card review (0 if first card review)
  # Same note as above about scaling
  df['time_since_first_review'] = (df['timestamp'] - df.groupby('card_id')['timestamp'].transform('first')).dt.total_seconds()

  # Scale to [0, 1]
  scaler = MinMaxScaler()
  normalized = scaler.fit_transform(df[["time_since_prev_review"]])
  df[["time_since_prev_review_minmax"]] = normalized

  # Scale to mean 0, std 1
  scaler = StandardScaler()
  normalized = scaler.fit_transform(df[['time_since_prev_review', 'time_since_first_review', 'secs_to_answer']])
  df[['time_since_prev_review_standard', 'time_since_first_review', 'time_to_answer']] = normalized

  # Drop 'timestamp'
  df = df.drop(columns=['timestamp', 'time_since_prev_review', 'secs_to_answer'])
  return df


df = main()
df.to_csv(output_file, index=False)