class Stack:
  def __init__(self):
    self.stack = []

  def insert(self,element):
    self.stack.append(element)

  def remove(self):
    if self.stack is None:
      return "stack is empty"
    else:
      return self.stack.pop()

  def length(self):
    return len(self.stack)

  def peek(self):
    return self.stack[-1]
#LINEAR SEARCH 
  '''def search(self,num):
    i=0
    while i <= len(self.stack):
      if self.stack[i]==num:
        return i
      i+=1
    return -1'''
# USING BINARY SEARCH COMPLEXITY GREATER THAN LINEAR SEARCH 
  def search(self,num):
    low =0 
    high =len(self.stack)-1
    result =-1
    while low <=high:
      mid = (low+high)//2
      if self.stack[mid]==num:
        return mid
      elif self.stack[mid]<num:
        low = mid+1
      else:
        high = mid-1
    return result

        

    i=0
    while i <= len(self.stack):
      if self.stack[i]==num:
        return i
      i+=1
    return -1

  def display(self):
    return self.stack

stack = Stack()
stack.insert(7)
stack.insert(8)
stack.insert(9)
stack.insert(10)
stack.insert(11)
print(stack.display())
stack.insert(12)
print(stack.display())
print(stack.peek())
print(stack.length())
num = 10
print(stack.search(num))