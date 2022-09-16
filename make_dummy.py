import csv

with open('data/Task_3_dataset/checkins_lessons_checkouts_training.csv', 'r') as inp, open('data/Task_3_dataset/tmp_training.csv', 'w') as out:
	reader = csv.reader(inp)
	header = next(reader)
	writer = csv.writer(out)
	writer.writerow(header)
	for row in reader:
		# print(row, row[2])
		if 1 < int(row[2]) < 5:
			writer.writerow(row)


