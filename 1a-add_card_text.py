"""
After the raw Anki export is cleaned up, this file extracts the card text from the Anki file and adds it to the CSV.

Note: this also limits the data to cloze deletion cards.

Input columns:
- card_id | timestamp | was_remembered | answer_score | secs_to_answer | review_type

Output columns:
- card_id | timestamp | was_remembered | answer_score | secs_to_answer | review_type | front | back

New column details:
front {str} -- the card text with the cloze being tested replaced with _____.
  e.g. The rectus muscles have lines of pull from _____ attachments to the _____ part of the eyeball
back {str} -- the clozes being tested, comma-separated.
  e.g. posterior, anterior
"""


import csv
import html
import re
from anki_export import ApkgReader


input_file = 'review_history.csv'
output_file = 'review_history_with_text.csv'
anki_file = 'decks.apkg'


def format_card_text(card_text, cloze_num):
  """Clean/format the card text and return the front and back.

  Arguments:
      card_text {str} -- The text of the card, with cloze deletions still in place
        e.g. "The {{c1::rectus muscles}} have lines of pull from {{c2::posterior}} attachments to the {{c2::anterior}} part of the eyeball"
      cloze_num {str} -- The cloze number being tested, indexed from 1
        e.g. "2"

  Returns:
      tuple -- 2-tuple of (front, back)

      front -- the card text with the cloze(s) being tested replaced with _____.
        e.g. "The rectus muscles have lines of pull from _____ attachments to the _____ part of the eyeball"
      back -- the clozes being tested (comma-separated if more than one)
        e.g. "posterior, anterior"
  """
  # Unescape HTML entities and clean up newlines
  card_text = html.unescape(card_text)
  card_text = re.sub(r'\n+', ' ', card_text).strip()

  # Append "extra" (clues; ignored for now)
  # extras = "".join(note_fields[1:])
  # extras = re.sub(r'<.*?>', '', extras)
  # extras = re.sub(r'<img.*?>', '', extras)

  # Replace the cloze being tested with _____.
  front = re.sub(r'{{c' + cloze_num + r'::(.*?)(::(.*?))?}}', lambda m: '_____ [{}]'.format(
      m.group(3)) if m.group(3) else '_____', card_text)
  front = re.sub(
       r'{{c\d+::(.*?)(::(.*?))?}}', r'\1', front)

  # After removing images, some cards lack sufficient cue power
  if front == "_____" or front == "This is _____":
      return None, None

  # For the 'Back', comma-separate the cloze values
  clozes = re.findall(r'{{c' + cloze_num + r'::(.*?)}}', card_text)
  # If a clue is present (indicated by ::), only keep the first part
  clozes = [cloze.split('::')[0] for cloze in clozes]
  back = ", ".join(clozes)

  return front, back


def main():
  # -------------------------
  # 1. get all card IDs from input CSV file
  # -------------------------
  with open(input_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header
    card_ids = {int(row[0]) for row in reader}

  # -------------------------
  # 2. extract and format text
  # -------------------------
  card_ids_to_text = {}  # {card_id: (front, back)}
  with ApkgReader(anki_file) as apkg:
    for card_id in card_ids:
      try:
        card_info = apkg.find_card_by_id(card_id)
      except Exception as e:
        continue

      # get note text, which is the card text with cloze deletions still in place
      # e.g. "This is a {{c1: cloze}} {{c2: deletion}} card."
      note_info = card_info['data']['note']
      note_fields = note_info['flds'].split('\x1f')
      card_text = note_fields[0]

      # limit to cloze deletion cards
      model_id = note_info['mid']
      model_info = apkg.find_model_by_id(model_id)
      if "cloze" not in model_info["name"].lower():
          continue

      # ord identifies which of the note cloze deletions this card is for, indexed from 0
      ord = card_info['ord']
      cloze_num = str(ord + 1)

      # get cardtext
      front, back = format_card_text(card_text, cloze_num)
      if front and back:
        card_ids_to_text[card_id] = (front, back)

  # -------------------------
  # 3. write output CSV file
  # same as input, but with front and back added
  # -------------------------
  with open(input_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    headers = next(reader)
    with open(output_file, 'w', newline='') as csvfile:
      writer = csv.writer(csvfile, delimiter=',',
                          quotechar='"', quoting=csv.QUOTE_MINIMAL)
      writer.writerow(headers + ['front', 'back'])
      for row in reader:
        card_id = int(row[0])
        if card_id not in card_ids_to_text:
          continue
        front, back = card_ids_to_text[card_id]
        writer.writerow(row + [front, back])


main()
