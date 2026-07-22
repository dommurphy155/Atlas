import json

# Reproduce the exact bytes from the test file line 128
with open('tests/test_coverage_gaps.py') as f:
    lines = f.readlines()

# First chunk (line 127)
line1 = lines[126].strip()
b1 = eval(line1[len('yield '):])
print('Chunk 1 bytes:', repr(b1))
jpart1 = b1[6:].strip()
try:
    j1 = json.loads(jpart1)
    print('Chunk 1 JSON OK:', j1)
    args1 = j1['choices'][0]['delta']['tool_calls'][0]['function']['arguments']
    print('Chunk 1 arguments:', repr(args1))
except Exception as e:
    print('Chunk 1 ERROR:', e)

# Second chunk (line 128)
line2 = lines[127].strip()
b2 = eval(line2[len('yield '):])
print()
print('Chunk 2 bytes:', repr(b2))
jpart2 = b2[6:].strip()
try:
    j2 = json.loads(jpart2)
    print('Chunk 2 JSON OK:', j2)
    args2 = j2['choices'][0]['delta']['tool_calls'][0]['function']['arguments']
    print('Chunk 2 arguments:', repr(args2))
except Exception as e:
    print('Chunk 2 ERROR:', e)

# Concatenate
combined = args1 + args2 if 'args1' in dir() and 'args2' in dir() else None
print()
print('Combined:', repr(combined))
try:
    print('Parsed:', json.loads(combined))
except Exception as e:
    print('Combined ERROR:', e)
