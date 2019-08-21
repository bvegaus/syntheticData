import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
plt.style.use('ggplot')

import xgboost as xgb

import pickle
import gc
import os
import sys

from keras import applications
from keras import backend as K
from keras import layers
from keras import models
from keras import optimizers
import tensorflow as tf

    
def get_data_batch(train, batch_size, seed=0):
    start_i = (batch_size * seed) % len(train)
    stop_i = start_i + batch_size
    shuffle_seed = (batch_size * seed) // len(train)
    np.random.seed(shuffle_seed)
    train_ix = np.random.choice( list(train.index), replace=False, size=len(train) ) # wasteful to shuffle every time
    train_ix = list(train_ix) + list(train_ix) # duplicate to cover ranges past the end of the set
    x = train.loc[ train_ix[ start_i: stop_i ] ].values
    
    return np.reshape(x, (batch_size, -1) )


def PlotData( x, g_z, data_cols, label_cols=[], seed=0, with_class=False, data_dim=2, save=False, prefix='' ):
    
    real_samples = pd.DataFrame(x, columns=data_cols+label_cols)
    gen_samples = pd.DataFrame(g_z, columns=data_cols+label_cols)
    
    f, axarr = plt.subplots(1, 2, figsize=(6,2) )
    if with_class:
        axarr[0].scatter( real_samples[data_cols[0]], real_samples[data_cols[1]], c=real_samples[label_cols[0]]/2 ) #, cmap='plasma'  )
        axarr[1].scatter( gen_samples[ data_cols[0]], gen_samples[ data_cols[1]], c=gen_samples[label_cols[0]]/2 ) #, cmap='plasma'  )
        

        
    else:
        axarr[0].scatter( real_samples[data_cols[0]], real_samples[data_cols[1]]) #, cmap='plasma'  )
        axarr[1].scatter( gen_samples[data_cols[0]], gen_samples[data_cols[1]]) #, cmap='plasma'  )
    axarr[0].set_title('real')
    axarr[1].set_title('generated')   
    axarr[0].set_ylabel(data_cols[1]) # Only add y label to left plot
    for a in axarr: a.set_xlabel(data_cols[0]) # Add x label to both plots
    axarr[1].set_xlim(axarr[0].get_xlim()), axarr[1].set_ylim(axarr[0].get_ylim()) # Use axes ranges from real data for generated data
    
        
    plt.show()

    

#### Functions to define the layers of the networks used in the 'define_models' functions below
    
    
def generator_network_w_label(x, labels, data_dim, label_dim, base_n_count): 
    x = layers.concatenate([x,labels])
    x = layers.Dense(base_n_count*1, activation='relu')(x) # 1
    x = layers.Dense(base_n_count*2, activation='relu')(x) # 2
    x = layers.Dense(base_n_count*4, activation='relu')(x)
    x = layers.Dense(data_dim)(x)    
    x = layers.concatenate([x,labels])
    return x
    
def discriminator_network(x, data_dim, base_n_count):
    x = layers.Dense(base_n_count*4, activation='relu')(x)
    x = layers.Dense(base_n_count*2, activation='relu')(x)
    x = layers.Dense(base_n_count, activation='relu')(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    return x
    


    
    
#### Functions to define the keras network models    

def define_models_CGAN(rand_dim, data_dim, label_dim, base_n_count, type=None):
    generator_input_tensor = layers.Input(shape=(rand_dim, ))
    labels_tensor = layers.Input(shape=(label_dim,)) # updated for class
    generated_image_tensor = generator_network_w_label(generator_input_tensor, labels_tensor, data_dim, label_dim, base_n_count) # updated for class

    generated_or_real_image_tensor = layers.Input(shape=(data_dim + label_dim,)) # updated for class
    
    discriminator_output = discriminator_network(generated_or_real_image_tensor, data_dim + label_dim, base_n_count) # updated for class

    generator_model = models.Model(inputs=[generator_input_tensor, labels_tensor], outputs=[generated_image_tensor], name='generator') # updated for class
    discriminator_model = models.Model(inputs=[generated_or_real_image_tensor],
                                       outputs=[discriminator_output],
                                       name='discriminator')

    combined_output = discriminator_model(generator_model([generator_input_tensor, labels_tensor])) # updated for class
    combined_model = models.Model(inputs=[generator_input_tensor, labels_tensor], outputs=[combined_output], name='combined') # updated for class
    
    return generator_model, discriminator_model, combined_model

   


#### Functions specific to the vanilla GAN architecture   
        
def training_steps_GAN(model_components):
    
    [ cache_prefix, with_class, starting_step,
                        train, data_cols, data_dim,
                        label_cols, label_dim,
                        generator_model, discriminator_model, combined_model,
                        rand_dim, nb_steps, batch_size, 
                        k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
                        data_dir, generator_model_path, discriminator_model_path, show,
                        combined_loss, disc_loss_generated, disc_loss_real, xgb_losses ] = model_components  
    
    for i in range(starting_step, starting_step+nb_steps):
        K.set_learning_phase(1) # 1 = train

        # train the discriminator
        z = np.random.normal(size=(batch_size, rand_dim))
        x = get_data_batch(train, batch_size, seed=i)
        

        if with_class:
                labels = x[:,-label_dim:]
                g_z = generator_model.predict([z, labels])
                
        else:
                g_z = generator_model.predict(z)
       
        d_l_r = discriminator_model.train_on_batch(x, np.random.uniform(low=0.999, high=1.0, size=batch_size)) # 0.7, 1.2 # GANs need noise to prevent loss going to zero
        d_l_g = discriminator_model.train_on_batch(g_z, np.random.uniform(low=0.0, high=0.001, size=batch_size)) # 0.0, 0.3 # GANs need noise to prevent loss going to zero

        disc_loss_real.append(d_l_r)
        disc_loss_generated.append(d_l_g)

      
        
        # train the generator

        z = np.random.normal(size=(batch_size, rand_dim))
        if with_class:
            loss = combined_model.train_on_batch([z, labels], np.random.uniform(low=0.999, high=1.0, size=batch_size)) # 0.7, 1.2 # GANs need noise to prevent loss going to zero
        else:
            loss = combined_model.train_on_batch(z, np.random.uniform(low=0.999, high=1.0, size=batch_size)) # 0.7, 1.2 # GANs need noise to prevent loss going to zero
        combined_loss.append(loss)

        if not i % log_interval:
            print('Step: {} of {}.'.format(i, starting_step + nb_steps))
            K.set_learning_phase(0) # 0 = test
                        
            # loss summaries      
            print( 'Losses: G, D Gen, D Real: {:.4f}, {:.4f}, {:.4f}'.format(combined_loss[-1], disc_loss_generated[-1], disc_loss_real[-1]) )
            print( 'D Real - D Gen: {:.4f}'.format(disc_loss_real[-1]-disc_loss_generated[-1]) )            

            
            if show:
                PlotData( x, g_z, data_cols, label_cols, seed=0, with_class=with_class, data_dim=data_dim, 
                            save=False, prefix= data_dir + cache_prefix + '_' + str(i) )
            
            # save model checkpoints
            model_checkpoint_base_name = data_dir + cache_prefix + '_{}_model_weights_step_{}.h5'
            generator_model.save_weights(model_checkpoint_base_name.format('generator', i))
            discriminator_model.save_weights(model_checkpoint_base_name.format('discriminator', i))

    
    return [combined_loss, disc_loss_generated, disc_loss_real]
    
def adversarial_training_GAN(arguments, train, data_cols, label_cols=[], seed=0, starting_step=0):

    [rand_dim, nb_steps, batch_size, 
             k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
            data_dir, generator_model_path, discriminator_model_path, loss_pickle_path, show ] = arguments
    
    np.random.seed(seed)     # set random seed
    
    data_dim = len(data_cols)
    print('data_dim: ', data_dim)
    print('data_cols: ', data_cols)
    
    label_dim = 0
    with_class = False
    if len(label_cols) > 0: 
        with_class = True
        label_dim = len(label_cols)
        print('label_dim: ', label_dim)
        print('label_cols: ', label_cols)
    
    # define network models
    
    K.set_learning_phase(1) # 1 = train
    
    if with_class:
        cache_prefix = 'CGAN'
        generator_model, discriminator_model, combined_model = define_models_CGAN(rand_dim, data_dim, label_dim, base_n_count)
    else:
        print("error")

    
    # compile models

    adam = optimizers.Adam(lr=learning_rate, beta_1=0.5, beta_2=0.9)

    generator_model.compile(optimizer=adam, loss='binary_crossentropy')
    discriminator_model.compile(optimizer=adam, loss='binary_crossentropy')
    discriminator_model.trainable = False
    combined_model.compile(optimizer=adam, loss='binary_crossentropy')
    
    if show:
        print(generator_model.summary())
        print(discriminator_model.summary())
        print(combined_model.summary())

    combined_loss, disc_loss_generated, disc_loss_real, xgb_losses = [], [], [], []
    
    if loss_pickle_path:
        print('Loading loss pickles')
        [combined_loss, disc_loss_generated, disc_loss_real, xgb_losses] = pickle.load(open(loss_pickle_path,'rb'))
    if generator_model_path:
        print('Loading generator model')
        generator_model.load_weights(generator_model_path, by_name=True)
    if discriminator_model_path:
        print('Loading discriminator model')
        discriminator_model.load_weights(discriminator_model_path, by_name=True)

    model_components = [ cache_prefix, with_class, starting_step,
                        train, data_cols, data_dim,
                        label_cols, label_dim,
                        generator_model, discriminator_model, combined_model,
                        rand_dim, nb_steps, batch_size, 
                        k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
                        data_dir, generator_model_path, discriminator_model_path, show,
                        combined_loss, disc_loss_generated, disc_loss_real, xgb_losses ]
        
    [combined_loss, disc_loss_generated, disc_loss_real] = training_steps_GAN(model_components)
        


