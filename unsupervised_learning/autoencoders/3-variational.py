#!/usr/bin/env python3
"""Creating a variational autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Function that creates a variational autoencoder

    input_dims is an integer containing the
        dimensions of the model input
    hidden_layers is a list containing the number of nodes
        for each hidden layer in the encoder, respectively
        the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions
        of the latent space representation

    Return: encoder, decoder, auto

    """
    input_encoder = keras.Input(shape=(input_dims, ))
    input_decoder = keras.Input(shape=(latent_dims, ))

    encoded = keras.layers.Dense(hidden_layers[0],
                                 activation='relu')(input_encoder)
    for enc in range(1, len(hidden_layers)):
        encoded = keras.layers.Dense(hidden_layers[enc],
                                     activation='relu')(encoded)

    z_mean = keras.layers.Dense(latent_dims, activation=None)(encoded)
    z_log_sigma = keras.layers.Dense(latent_dims, activation=None)(encoded)

    def sample_z(args):
        """
        Sampling function
        """
        mu, sigma = args
        batch = keras.backend.shape(mu)[0]
        dim = keras.backend.int_shape(mu)[1]
        eps = keras.backend.random_normal(shape=(batch, dim))
        return mu + keras.backend.exp(sigma / 2) * eps

    z = keras.layers.Lambda(sample_z,
                            output_shape=(latent_dims,))([z_mean, z_log_sigma])

    encoder = keras.Model(inputs=input_encoder,
                          outputs=[z, z_mean, z_log_sigma])

    decoded = keras.layers.Dense(hidden_layers[-1],
                                 activation='relu')(input_decoder)
    for dec in range(len(hidden_layers) - 2, -1, -1):
        decoded = keras.layers.Dense(hidden_layers[dec],
                                     activation='relu')(decoded)
    last = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(inputs=input_decoder, outputs=last)

    # Link encoder outputs through decoder
    auto_outputs = decoder(encoder(input_encoder)[0])
    auto = keras.Model(inputs=input_encoder, outputs=auto_outputs)

    def vae_loss(x, x_decoded_mean):
        """variational autoencoder loss function"""
        # Compute reconstruction loss manually using raw math to avoid Keras backend auto-averaging
        recon_loss = -keras.backend.sum(
            x * keras.backend.log(keras.backend.epsilon() + x_decoded_mean) +
            (1 - x) * keras.backend.log(keras.backend.epsilon() + 1 - x_decoded_mean),
            axis=-1
        )
        
        # Compute KL divergence loss
        kl_loss = - 0.5 * keras.backend.sum(
            1 + z_log_sigma - keras.backend.square(z_mean) - keras.backend.exp(
                z_log_sigma), axis=-1)
        
        # Return the final mean over the batch instances
        return keras.backend.mean(recon_loss + kl_loss)
