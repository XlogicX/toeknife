import binascii
import argparse

class nonprefix:
	'Structure for non-prefix codes, supposed to be huffman, but flexible...'
	'alphabet: all symbols for table'
	'bitlengths: amount of bits for each symbol in same order as alphabet'
	'table: the generated non-prefix code table, built be construct()'

	def __init__(self,alphabet,bitlengths):
		self.alphabet = alphabet
		self.bitlengths = bitlengths

		# Edge case for huffman table for empty distance table...this could be a bad idea,
		# but it fixes more than it breaks
		if not bitlengths:
			self.bitlengths = [1]

	def construct(self):
		# Construction based on algorithm from RFC 1951
		self.table = {}
		# Step 1: Count the number of codes for each code length.  Let bl_count[N] be the number of codes of length N, N >= 1.

		# Initiliaze bl_count array with highest bitcount found in bitlengths
		MAX_BITS = max(self.bitlengths)
		bl_count = [0 for i in range(MAX_BITS+1)]
		for i in self.bitlengths:	# for each of the bit lengths
			bl_count[i] += 1	# 	tally them up

		# Find if it is oversubscribed (This might never be reached; there are other checks and balances)
		left = 1
		for i in range(1,MAX_BITS+1):
			left <<= 1
			left -= bl_count[i]
			if (left < 0):
				return left

		# Step 2: Find the numerical value of the smallest code for each
		code = 0
		bl_count[0] = 0
		next_code = [0 for i in range(MAX_BITS+1)]
		for i in range(1,MAX_BITS+1):
			code = (code + bl_count[i-1]) << 1
			next_code[i] = code

		# Step 3: Assign numerical values to all codes, using consecutive values for all codes of the same length with the base
		# 	values determined at step 2. Codes that are never used (which have a bit length of zero) must not be assigned a value.
		tree = [0 for i in range(len(self.alphabet))]
		for n in range(len(self.alphabet)):
			length = self.bitlengths[n]
			if length != 0:
				tree[n] = next_code[length]
				next_code[length] += 1

		for idx,i in enumerate(tree):
			code = '{num:0{width}b}'.format(num=i,width=self.bitlengths[idx])
			if code == str(0):
				self.table[self.alphabet[idx]] = '2'	# Hacky way to encode a bitlength of zero (because 2 isn't binary)
			else:
				self.table[self.alphabet[idx]] = code

class bitstream:
	# Builds up a large bitstream ordered in a sane way. Initilize with binary data
	def __init__(self, stream):
		self.stream = stream

		# Creates long string of 1's and 0's and reverses them
		self.bits = ''
		for byte in stream:
			self.bits += '{num:0{width}b}'.format(num=byte, width=8)[::-1]

	# Removes n/rep number of bits
	def pop(self,rep):
		self.bits = self.bits[rep:]

	# Get bits without popping (peek)
	def fetchbits(self,bits):
		return self.bits[0:bits]

	# Get n number of bits and pops them out
	def extractbits(self,bits):
		ebits = self.bits[0:bits]
		self.bits = self.bits[bits:]
		return ebits	

def huffsearch(table,bits):
	# We don't know how many bits the next token will be, so we search the huffman table for the first one that matches
	for i in range(32):								# keep growing bitlength each time we try until match
		if bits.fetchbits(i) in table.values():		# If there is a match
			token = list(table.keys())[list(table.values()).index(bits.fetchbits(i))]	# grab it
			bits.pop(i)																	# pop it
			return token																# return it
	else:
		pass

def getbits(bits,n):
	# We do know how many bits we need, as we are already aligned on extra bits or a distance component of a token
	partial = bits.bits[0:n]
	bits.pop(n)
	return partial

def carve(bits,guesses):
	length_bases = [3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258]
	extra_length = [0,0,0,0,0,0,0,0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,  5,  5,  5,  5,  0]
	distance_bases = [1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577,32768];
	extra_distance = [0,0,0,0,1,1,2,2, 3, 3, 4, 4, 5, 5, 6,  6,  7,  7,  8,  8,  9,   9,   10,  10,  11,  11,  12,  12,   13,   13]
	buffer = b''
	symbolbuffer = ''
	length = 0
	distance = 0
	ldcount = 0
	try:
		while(1):
			token = huffsearch(huff.table,bits)
			if token == None:
				break
			if int(token) < 256:
				# Its a 'literal'
				if (int(token) > 31 and int(token) < 127):
					buffer += bytes(chr(int(token)), 'utf-8')
					symbolbuffer += chr(int(token))
				else:
					buffer += binascii.unhexlify("{0:0{1}x}".format(int(token),2))
					symbolbuffer += hex(int(token))				
			elif int(token) == 256:
				symbolbuffer += '[EOB]'
			else:
				# It's a Length-Distance

				# Process Length Details
				length_offset = int(token) - 257					# Index into length bases
				length_base = length_bases[length_offset]			# The length before adding extra length (if needed)
				extra_length_bits = extra_length[length_offset]		# Calculate extra bits needed for more length
				if extra_length_bits == 0:
					length = int(length_base)
				else:
					extra = getbits(bits,extra_length_bits)[::-1]
					full_length = length_base + int(extra, 2)
					length = int(full_length)
				# Process Distance Details
				distbits = huffsearch(disthuff.table,bits)
				distance_offset = int(distbits)							# Index into dist bases
				distance_base = distance_bases[distance_offset]			# The distance before adding extra distance (if needed)
				extra_distance_bits = extra_distance[distance_offset]		# Calculate extra bits needed for more length
				if extra_distance_bits == 0:
					distance = int(distance_base)
				else:
					extra = getbits(bits,extra_distance_bits)[::-1]
					full_distance = distance_base + int(extra, 2)
					distance = int(full_distance)

				# If the length-distance goes back farther than our current buffer
				if distance > len(buffer):
					# If we have a user-supplied data to use for that data
					if str(ldcount) in guesses:
						prepend = distance - len(buffer)		# how much farther back do we go in the buffer based on start of buffer
						# If the buffer goes farther back than the length of the guess
						if len(guesses[str(ldcount)]) <= prepend-len(guesses[str(ldcount)]):
							# Then the buffer is the guess + filled ?'s, + old buffer + guess
							buffer = bytes(guesses[str(ldcount)], 'utf-8') + bytes(('?'*int(prepend-len(guesses[str(ldcount)]))),'utf-8') + buffer + bytes(guesses[str(ldcount)],'utf-8')
						else:
							# Otherwise it is exact or partial, same as above without the fill and potentially partial slicing
							buffer = bytes(guesses[str(ldcount)][0:prepend], 'utf-8') + buffer + bytes(guesses[str(ldcount)],'utf-8')
					# If we don't have a guess
					else:
						# Just fill with ?'s
						buffer += bytes('?' * length, 'utf-8')

					# Populate the tokenized (non-lit-replace) version	
					symbolbuffer += '[{}:{},{}]'.format(ldcount,length,distance)

				# Otherwise the length-distance pair is within our buffer (and is therefore known)
				else:
					# Add Symbol Form
					symbolbuffer += '[{}:{},{}]'.format(ldcount,length,distance)
					# If the length is actually larger than the distance it goes back (it's looping)
					while (length >= distance):
						buffer += buffer[len(buffer)-distance:len(buffer)-distance+length]
						length -= distance
					# Either we are done with the looping part, or we just didn't have a distance smaller than length,
					# either way, process what's left
					else:
						buffer += buffer[len(buffer)-distance:len(buffer)-distance+length]

				ldcount += 1
		return(buffer,symbolbuffer)
	except:
		pass

def builddynhuff(hlit,table,huffstream):
	bitlengths = [0] * hlit								# Init list with of hlit size with 0's
	i = 0
	while i < hlit:										# for hlit times
		huffcode = int(huffsearch(table,huffstream))	# get the code
		if huffcode < 16:								# if less than 16, then take number at face value
			bitlengths[i] = huffcode
			i += 1
		if huffcode == 18:								# code '18' is 11-138 zeros
			zeros = int(huffstream.extractbits(7)[::-1],2) + 11		# calculate how many zeros
			for j in range(zeros):									# and place them in bitlengths table
				bitlengths[i] = 0
				i += 1
		if huffcode == 17:
			zeros = int(huffstream.extractbits(3)[::-1],2) + 3	# code '17' is 3-10 zeros
			for j in range(zeros):								# calculate how many zeros
				bitlengths[i] = 0								# and place them in bitlengths table
				i += 1
	return bitlengths

def buildtables():
	if args.dynamic:
		f = open(args.dynamic, 'rb')
		dhuffbin = f.read()					# Get huffman table as file data fragment
		huffstream = bitstream(dhuffbin)	# Get it as bitstream
		huffstream.pop(1)					# Pop off the 'last' field
		header = huffstream.extractbits(2)	# extract Mode field
		if header != '01':	# Actually equals 2/dyn if interpreted reversed...
			print("The start of fragment does not represent Dynamic Huffman Mode")
			quit()
		# Get Literal/Length, Distance, and Code Length codes
		hlit = int(huffstream.extractbits(5)[::-1], 2)+257	# How many literals and lengths there will be
		hdist = int(huffstream.extractbits(5)[::-1], 2)+1	# How many distances
		hclen = int(huffstream.extractbits(4)[::-1], 2)+4	# how many short-hand codes we need to encode that table
		# Building those short-hand codes (max of 19)
		codelengthcodes = [0] * 19	# 0 init
		# Order specified by rfc1951
		codelengthorder = ['16','17','18','0','8','7','9','6','10','5','11','4','12','3','13','2','14','1','15']
		codelengthorder = codelengthorder[0:hclen]	# Only use the codes specified by HCLEN
		# Parse through the code length codes
		for idx,code in enumerate(codelengthorder):
			codelengthcodes[int(code)] = int(huffstream.extractbits(3)[::-1],2)	# Extract bits and append integer form of code
		codehuff = nonprefix(list(range(0,19)),codelengthcodes)					# Construct table with sequential order
		codehuff.construct()

		# Create Dynamic Huffman Table For Lits and Lengths and Distances
		dynbitlengths = builddynhuff(hlit,codehuff.table,huffstream)
		dynbitlengths_dist = builddynhuff(hdist,codehuff.table,huffstream)

		alphabet = []
		for i in range (0,len(dynbitlengths)):
			alphabet.append(str(i))
		huff = nonprefix(alphabet,dynbitlengths)
		huff.construct()

		alphabet = []
		for i in range (0,len(dynbitlengths_dist)):
			alphabet.append(str(i))
		disthuff= nonprefix(alphabet,dynbitlengths_dist)
		disthuff.construct()
	else:
		# Building Huffman Table for Fixed mode, the predefined bitlengths below is enough information to generate this
		alphabet = []
		for i in range (0,288):
			alphabet.append(str(i))
			bitlengths = [8] * 144 + [9] * 112 + [7] * 24 + [8] * 8
		huff = nonprefix(alphabet,bitlengths)
		huff.construct()

		alphabet = []
		for i in range (0,29):
			alphabet.append(str(i))
			bitlengths = [5] * 29
		disthuff = nonprefix(alphabet,bitlengths)
		disthuff.construct()

	return(huff,disthuff)

def processguesses(guessdict):
	# Get and parse guess file
	# Each line is formated by the index (zero base) of the guess and the actual text of the guess, ie:
	#	5:sometext
	# The example above would be a guess for the 6th unknown token with a guess of 'someguess'
	with open (args.guesses, "r") as guesses:
		for guess in guesses:
			guessdict[guess.replace("\n", "").split(':')[0]] = guess.replace("\n", "").split(':')[1]
	return guessdict

def presentresults():
	print("Recovered Data: {}".format(output))
	print("Symbol Form:    {}".format(symboloutput))
	if args.table:
		print("Literal/Lengths:")
		literallist = []
		for token in huff.table:
			if huff.table[token] != '2':  # If valid value in huffman table
				if int(token) < 256:
					if (int(token) > 31 and int(token) < 127):
						literallist.append(chr(int(token)))
					else:
						literallist.append(hex(int(token)))				
				elif int(token) > 256:
						literallist.append('L-{}'.format(token))
		print(literallist)
		print("Distances:")
		distlist = []
		for token in disthuff.table:
			if disthuff.table[token] != '2':  # If valid value in huffman table
				distlist.append(token)
		print(distlist)

parser = argparse.ArgumentParser(prog='ToeKnife')
parser.add_argument('--window', help='How many bytes of sample decompressed data to present to you (default is 20)', default=20, type=int)
parser.add_argument('--skip', help='How many bits to pop from buffer for alignment (default 0)', default=0, type=int)
parser.add_argument('--file', help='Filename of the file that contains the compressed fragment', type=str)
parser.add_argument('--data', help='ASCIIHex representation of the filedata right on the commandline', type=str)
parser.add_argument('--guesses', help='Guessfile for partial data refered to by length-distance pairs', type=str)
parser.add_argument('--dynamic', help='provide file with initial data fragment of dynamic block', type=str)
parser.add_argument('--table', help='Dumps Huffman table to illustrate which characters are valid (for potential guesses)', action="store_true")
args = parser.parse_args()

# Get Data
if args.file:
	f = open(args.file, 'rb')
	sampledata = f.read()
elif args.data:
	sampledata = binascii.unhexlify(args.data)
else:
	print("You must supply a compressed fragment via the --file --data argument")
	quit()

# Process Guesses if provided
guessdict = {}
if args.guesses:
	guessdict = processguesses(guessdict)

huff, disthuff = buildtables()	# Build dynamic huffman table or build known fixed table if no dynamic huff

# Main Loop
for i in range(args.skip,26):		# Keep trying to pop bits out until things look right (26 tries)
	bits = bitstream(sampledata)				# Get bits (in proper orientation) of input data
	bits.pop(i)									# pop another bit out until alignment is correct (starts at 0)
	try:
		output,symboloutput = carve(bits,guessdict)	# Decompress
	except:
		output,symboloutput = 'INVALID','INVALID'

	# Instead of using hueristics, ask the user if the auditioned data looks right
	print(output[0:args.window])				# Just prints sample of data (not entire thing)
	tryagain = input("Does this look right?")
	if 'y' in tryagain.lower():
		break
	if i == 25:						# Longest theoretical bit pattern (length-distance token with max extra bits)
		print("BOTCHED TOE!!!")
	print()

# If it looked right, the whole thing will be printed
presentresults()
