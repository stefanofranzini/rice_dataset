# rice_dataset

Analysis of the Kaggle rice dataset

This repo contains a simple deep classifier for the rice dataset found on Kaggle (please, download separately).
Alongside the classifier I also wrote an autoencoder for dimensional reduction. Used in the naive way the embedded
points (in 2 dimensions) will display rotational symmetry owning to the different orientations of the rice seeds in
the dataset.
So I decided to create a second dataset by orienting all seeds so that their major axis goes along the diagonal of
the image. I then trained the autoencoder to learn the mapping between the original randomly oriented seed and the
rotated one.
By doing this the embedded space is regularized with respect to rotations and does not display the original symmetry,
allowing for better identification of the different rice families.
