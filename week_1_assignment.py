# LOWER TRIANGLE IN STAR PATTERN

n=5
for i in range(n+1):
    print("*"*i)
print()


# UPPER TRIANGLE IN STAR PATTERN

n=5

for i in range(n):
  print("*"*(n))
  n-=1
print()


# PYRAMID TRIANGLE IN STAR PATTERN

n=5
for i in range(1,n+1):
  print(' '*(n-i)+"*"*(2*i-1))
