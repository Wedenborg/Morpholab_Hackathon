## Eval team thoughts: 

### Method

Whatever method we come up with gives us at each time point the tuple of $(c_t, g_t)$ where c_t are the coordinates, locations of the cells to do sequencing on and g_t are the genes we choose to select. (this is a sequential experimental design view of the method). At each timestep t, the method gives the "design" of the next experiment. 

### Benchmark (1)

We think that whatever our method results in, should give us information to classify into different celltypes, like with (foregut, midgut, liver, pancreas etc.). 

Which we can do by finding a dataset that is not the one we train on and create a benchmark of embryo-wide aggregated counts from the representative genes. 

The method then should be able to predict the proportions of the celltypes that somehow correspond to distribution of truth we find in other data. 

