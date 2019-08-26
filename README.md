# Workspace_partitioner
Partition a given workspace to laser adapted regions

## Defining Workspace
The workspace is defined in a json file which contains all Lines, vertices and obstacles defined as follows:


#### Lines
```
"Line":
      {
        "x1": 0.0,
        "y1": 0.0,
        "x2": 2.5,
        "y2": 0.0,
        "orientation": "H",
        "index": 0
      }
```
where (x,y) are the coordinates of the two end points of the line and orientation is ether `H`, `V` or `I`(inclined)

#### Vertices

``` 
"vertex":
      {
        "L1": 0,
        "L2": 4,
        "x": 0.0, 
        "y": 0.0, 
        "angle_lb": 0.0, 
        "angle_ub": 90.0
      }
      
```

A vertex is an intersection of two lines with indices `L1` and `L2`, a point of intersection `(x,y)`, and a lower and upper bounds on the angle (facing the free space) between the two lines.

#### Obstacles

```
"obstacle":
      {
        "vertices":[
          [0,3.5],[0,6],[4,6],[2.5,3.5]
        ]
      }
```

Obstacle is defined as a set of vertices.

## Running the partition algorithm

```python state_machine $NUM_VERTICES $NUM_LASERS```

