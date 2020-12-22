# bookcipher


Hi there! Sorry, but this readme isn't really up to date yet.
The gist of what I'm doing with this project is that I'm attempting
to decrypt a book cipher (cipher -> plaintext, key), with no (or very little)
prior knowledge about key or plaintext used.

Currently, I have a working prototype that given N letters of cipher and N-1
letters of known plaintext/key, it can decrypt the Nth letter of cipher into plaintext.
We then recursively feed this prediction into the model to decrypt the N+1th, N+2nd, N+3rd, character
while still only having knowledge about the first N-1 indices.


# Example: Known start
Say that the cipher text C is known, and the first N letters of the message M are known.

C = `tigzifzomabpbbfijfvtsreijaetwqrdouwcrpdaslizlbmvhjmrtogzcjgya`<br/>
M = `theweatherreporttodaywillbesunnyinothernews------------------`<br/>
K = `abcdefghijklmnopqrstuvwxyzabcdefghijklmnopq------------------`<br/>

itisalmostnewyears

To make this case simple, I've let K be the alphabet repeated, though in practice it should
be a string of ENGLISH words.

Given the first 43 characters, the current iteration of the program can decrypt the 44th character,
recovering M[44] = i, and K[44] = r.
We can then update our knowledge base:

C = `tigzifzomabpbbfijfvtsreijaetwqrdouwcrpdaslizlbmvhjmrtogzcjgya`<br/>
M = `theweatherreporttodaywillbesunnyinothernewsi-----------------`<br/>
K = `abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqr-----------------`<br/>

Then we may recover M[45] = t, K[45] = s. Once again, we may update our knowledge base and iterate:

C = `tigzifzomabpbbfijfvtsreijaetwqrdouwcrpdaslizlbmvhjmrtogzcjgya`<br/>
M = `theweatherreporttodaywillbesunnyinothernewsit----------------`<br/>
K = `abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrs----------------`<br/>

By repeated application, we may eventually decrypt the entire stream to find that our message and key decrypt as follows!

C = `tigzifzomabpbbfijfvtsreijaetwqrdouwcrpdaslizlbmvhjmrtogzcjgya`<br/>
M = `theweatherreporttodaywillbesunnyinothernewsitisalmostnewyears`<br/>
K = `abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghi`<br/>

Woohoo! we have decrypted the entire message, given only the start, and little knowledge of the key or message 
(and no knowledge of the parts we are decrypting)

# Caveats:
It's obvious that we may not always be able to know the first N letters of the message.
I'm currently working on creating encodings that would remove this dependency,
thought the amount of data required to accurately model this is quite large,
and my computer is quite small. I anticipate that I will be able to do this, but the
simple "throw more data at it" idea won't work for me so I will have to come up with a better solution.

Another possible problem is that the program gets 'lost'. Given the nature of the problem, if the algorithm
makes a big enough mistake somewhere in the middle, everything after that will be garbage. We'll also need
to protect against noise (ie: the algorithm makes a small mistake, but recovers, or perhaps the message has a tpyo in it)
