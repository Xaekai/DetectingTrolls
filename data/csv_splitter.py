#Credit @AlexDel, some modifications made
#https://gist.github.com/AlexDel
import csv

filename = 'non_bot.csv'
chunksize = 200000
chunksdir = ''

f = open(filename, 'r')
myreader = csv.reader(f, delimiter=',')

headings = next(myreader)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]

rows = [i for i in myreader]
chunked_rows = [chunk for chunk in chunks(rows,chunksize)]

for i in range(len(chunked_rows)):
    c = open('non_bot_chunk_' + str(i) + '.csv', 'w+')
    mywriter = csv.writer(c, delimiter=',')
    for chunk_row in chunked_rows[i]:
         mywriter.writerow(chunk_row)
    c.close()
f.close()
