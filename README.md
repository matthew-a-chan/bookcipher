# bookcipher


Hi there! Sorry, but this readme isn't really up to date yet.
The gist of what I'm doing with this project is that I'm attempting
to decrypt a book cipher (cipher -> plaintext, key), with no (or very little)
prior knowledge about key or plaintext used.

Currently, I have a working prototype that given N letters of cipher and N-1
letters of known plaintext/key, it can decrypt the Nth letter of cipher into plaintext.
We then recursively feed this prediction into the model to find the N+1th character...

IE: if the cipher text is akwlqwlktnsoivwkejrkwejwkfjkewlnlvcxiqejrlkqwnlwnefn, and we know the message starts with:
"weatherreporttodaywillbesunnyinothernews------", we should theoretically be able to
continue the decryption process and recover more information.