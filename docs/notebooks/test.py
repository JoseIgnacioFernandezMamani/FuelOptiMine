from turtle import*
shape('classic')
speed(30)

def star(length):
   for i in range(5):
      forward(length)
      right(180-36)
      
star(200)
done()