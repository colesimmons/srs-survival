import csv
import re
import time
from collections import defaultdict
from anki_export import ApkgReader
import html
from datetime import datetime

# Read the CSV file
input_file = 'input.csv'
output_file = 'output.csv'
anki_file = 'decks.apkg'

# Data structure to store the data
data = defaultdict(list)

def extract():
  added_review_ids = set()

  # Extract Anki cards
  with ApkgReader(anki_file) as apkg:

    # Read the CSV file
    with open(input_file, 'r') as csvfile:
      reader = csv.reader(csvfile)
      # Skip header
      next(reader)

      # Write the preprocessed data to a new CSV file
      with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Writer the output header
         # header = ['card_id', 'timestamp', 'front', 'back', 'wasRemembered', 'difficulty', 'timeToAnswer']
        header = ['card_id', 'timestamp', 'was_remembered', 'time_to_answer']
        writer.writerow(header)

        # Process each row, which can have up to 2 reviews
        # Structure is:
        # Date1,Answer1,Interval,Ease factor,Time to answer1,Review type1,Date2,Answer2,Time to answer2,Review type2,Card ID
        for row in reader:
          try:
            card_id = row[-1]

            # ----------------- CARD INFO -----------------
            card_info = apkg.find_card_by_id(card_id)

            # card id : the epoch milliseconds of when the card was created
            _id = card_info['id']

            # notes.id
            nid = card_info['nid']

            # deck id (available in col table)
            did = card_info['did']

            # ordinal : identifies which of the card templates or cloze deletions it corresponds to
            #   for card templates, valid values are from 0 to num templates - 1
            #   for cloze deletions, valid values are from 0 to max cloze index - 1 (they're 0 indexed despite the first being called `c1`)
            # the position of the template of the model; it matches the template's ord field.
            ord = card_info['ord']

            # other fields available on card:
            #  mod : the epoch milliseconds of when the card was last modified
            #  usn : update sequence number : used to figure out diffs when syncing.
            #  type : 0=new, 1=learning, 2=review, 3=relearning
            #  queue : int from -3 - 4 (suspended, user buried, sched buried, user buried, new, learning, review)
            #  due : due is used differently for different card types, but used to find next time card is due
            #  ivl : interval (used in SRS algorithm). Negative = seconds, positive = days
            #  factor : The ease factor of the card in permille (parts per thousand). If the ease factor is 2500, the card’s interval will be multiplied by 2.5 the next time you press Good.
            #  reps : # of reviews
            #  lapses : the number of times the card went from a "was answered correctly" to "was answered incorrectly" state
            #  left : the number of reviews left before the card graduates from learning
            #  odue : original due: only used when the card is currently in filtered deck
            #  odid : original did: only used when the card is currently in filtered deck
            #  flags : an integer. This integer mod 8 represents a "flag", which can be see in browser and while reviewing a note. Red 1, Orange 2, Green 3, Blue 4, no flag: 0. This integer divided by 8 represents currently nothing
            #  data["deck"] - deck info

            # ----------------- NOTE INFO -----------------
            note_info = card_info['data']['note']

            # mid : model id
            model_id = note_info['mid']

            # flds : the values of the fields in this note. separated by 0x1f (31) character.
            note_fields = note_info['flds'].split('\x1f')

            # other fields available on the note
            #   guid : used for syncing
            #   mod : modification timestamp
            #   usn : updated sequence number : used to figure out diffs when syncing.
            #   tags : space-separated string of tags. includes space at the beginning and end, for LIKE "% tag %" queries.
            #   sfld : sort field: used for quick sorting and duplicate check.
            #       The sort field is an integer so that when users are sorting on a field that contains only numbers, they are sorted in numeric instead of lexical order.
            #       Text is stored in this integer field.
            #   csum : field checksum used for duplicate check.
            #   flags : unused
            #   data : unused

            # ----------------- MODEL INFO -----------------
            model_info = apkg.find_model_by_id(model_id)
            # css : CSS, shared for all templates
            # did : Long specifying the id of the deck that cards are added to by default
            # flds : JSONArray containing object for each field in the model as follows:
            #   eg. [{
            #       font : "display font",
            #       media : "array of media. appears to be unused",
            #       name : "field name",
            #       ord : "ordinal of the field - goes from 0 to num fields -1",
            #       rtl : "boolean, right-to-left script",
            #       size : "font size",
            #       sticky : "sticky fields retain the value that was last added when adding new notes"
            #   }]
            # id : model ID, matches notes.mid
            # latexPost : String added to end of LaTeX expressions (usually \\end{document})"
            # latexPre : preamble for LaTeX expressions
            # mod : modification time in seconds
            # name : model name
            # req : unused in modern clients. May exist for backwards compatibility.
            # sortf : Integer specifying which field is used for sorting in the browser
            # tags : Anki saves the tags of the last added note to the current model, use an empty array []
            # tmpls : JSONArray containing object of CardTemplate for each card in model
            #   eg. [{
            #       afmt : answer template string,
            #       bafmt : browser answer format: used for displaying answer in browser,
            #       bqfmt : browser question format: used for displaying question in browser,
            #       did : deck override (null by default),
            #       name : template name,
            #       ord : template number, see flds,
            #       qfmt : question format string
            #   }]
            # type : Integer specifying what type of model. 0 for standard, 1 for cloze
            # usn : Update sequence number: used in same way as other usn vales in db
            # vers : Legacy version number (unused), use an empty array []
            if "cloze" not in model_info["name"].lower():
                continue

            # ----------------- FORMAT CARD TEXT FOR CSV -----------------
            # Remove any HTML and images
            card_text = note_fields[0]
            card_text = re.sub(r'<.*?>', '', card_text)
            card_text = re.sub(r'<img.*?>', '', card_text)

            # Clean up newlines
            card_text = re.sub(r'\n+', ' ', card_text).strip()

            # Unescape HTML entities
            card_text = html.unescape(card_text)

            # Ignored for now
            extras = "".join(note_fields[1:])
            extras = re.sub(r'<.*?>', '', extras)
            extras = re.sub(r'<img.*?>', '', extras)

            # Get the cloze number for this card
            cloze_num = str(ord + 1)

            # Replace the cloze being tested with _____.
            # If there is a clue, include it as [clue].
            front = re.sub(r'{{c' + cloze_num + r'::(.*?)(::(.*?))?}}', lambda m: '_____ [{}]'.format(
                m.group(3)) if m.group(3) else '_____', card_text)
            front = re.sub(
                r'{{c\d+::(.*?)(::(.*?))?}}', r'\1', front)

            # After removing images, some cards lack sufficient cue power
            if front == "_____" or front == "This is _____":
              continue

            # For the 'Back', comma-separate the cloze values
            clozes = re.findall(r'{{c' + cloze_num + r'::(.*?)}}', card_text)
            # If a clue is present (indicated by ::), only keep the first part
            clozes = [cloze.split('::')[0] for cloze in clozes]
            back = ", ".join(clozes)

            # ----------------- WRITE REVIEWS TO CSV -----------------

            # Each row contains information about 2 subsequent reviews.
            # In theory, we'll only add review1 when it is the first review of a card.
            # Otherwise, it'll be covered by a previous row's review2.

            # Add data for each review
            review1 = {
                "at": datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S"),
                "answer": int(row[1]),
                "timeToAnswer": float(row[4]),
                "reviewType": row[5],
            }
            review_1_id = card_id + row[0]

            # IGNORING (duplicated on subsequent row unless is last row for card)
            review2 = {
                "at": datetime.strptime(row[6], "%Y-%m-%d %H:%M:%S"),
                "answer": int(row[7]),
                "timeToAnswer": float(row[8]),
                "reviewType": row[9],
            }
            review_2_id = card_id + row[6]

            reviews = [(review_1_id, review1), (review_2_id, review2)]

            reviews_to_add = [r for r in reviews if r[0] not in added_review_ids]

            # Convert answers to wasRemembered and difficulty
            for _id, review in reviews_to_add:
                was_remembered = 1 if review["answer"] != 1 else 0

                if review["answer"] == 4:  # Easy
                  difficulty = 1
                elif review["answer"] == 3:  # Good
                  difficulty = 2
                elif review["answer"] == 2:  # Hard
                  difficulty = 3
                else:  # Again (forgotten)
                  difficulty = 4

                at = review["at"]
                time_to_answer = review["timeToAnswer"]
                was_remembered = was_remembered
                difficulty = difficulty

                # Write the card data to the CSV
                # row = [card_id, at, front, back, was_remembered, difficulty, time_to_answer ]
                row = [card_id, at, was_remembered, time_to_answer ]
                writer.writerow(row)
                added_review_ids.add(_id)

          except Exception as inst:
            print(card_id)
            print(type(inst))
            print(inst.args)
            print(inst)
            continue
