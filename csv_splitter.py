#Credit @AlexDel, some modifications made
#https://gist.github.com/AlexDel
import csv
import sys

print(sys.argv[1])


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]
chunksize = 10000
chunksdir = ''

for arg in sys.argv[1:]:
	filename = arg
	f = open(filename, 'r', encoding="utf8")
	myreader = csv.reader(f, delimiter=',')

	headings = next(myreader)


	rows = [i for i in myreader]
	chunked_rows = [chunk for chunk in chunks(rows,chunksize)]

	for i in range(len(chunked_rows)):
		c = open(filename.split(".")[0] + '_chunk_' + str(i) + '.csv', 'w+', encoding="utf8")
		mywriter = csv.writer(c, delimiter=',')
		for chunk_row in chunked_rows[i]:
			 mywriter.writerow(chunk_row)
		c.close()
	f.close()



