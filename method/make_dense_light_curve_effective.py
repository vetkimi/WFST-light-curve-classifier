class make_dense_light_curve:
    def __init__(self,light_curve):
        self.light_curve = light_curve
        
        
    #def fit_gps(self):
    def get_light_curve(self):
        import numpy as np
        mjd = []
        flux = []
        flux_err = [] 
        
        for light_curves in self.light_curve:
            mjd_ = np.array(light_curves["mjd"])
            flux_ = np.array(light_curves["flux"])
            flux_err_ = np.array(light_curves["flux_err"])
            mjd.append(mjd_)
            flux.append(flux_)
            flux_err.append(flux_err_)
        return (mjd,flux,flux_err)
    
    
    #def concatenate_lc(self, light_curve_info, is_simulation=True):
    def concatenate_lc(self, light_curve_info):
        """
        parameter: light_info format: [mjd, flux, flux_err]
                   mjd, flux, flux_err contains multi-band light information
        """
        import numpy as np
        central_wave_length = [3570.0,4767.0,6215.0,7545.0,8708.0,10040.0]
        central_wave_length = np.array(central_wave_length)
        #light_curve_info=self.get_light_curve()

        mjd = light_curve_info[0]
        flux = light_curve_info[1]
        flux_err = light_curve_info[2]
        
        # concatnate different band lightcurves
        """
        mjd_stack = np.hstack([mjd[0],mjd[1],
                                   mjd[2],mjd[3],
                                   mjd[4],mjd[5]])
            
        flux_stack = np.hstack([flux[0],flux[1],
                                    flux[2],flux[3],
                                    flux[4],flux[5]])
            
        flux_err_stack = np.hstack([flux_err[0],flux_err[1],
                                        flux_err[2],flux_err[3],
                                        flux_err[4],flux_err[5]])
            
        wavelength_stack = np.hstack([np.ones(shape=mjd[0].shape)*central_wave_length[0],
                                      np.ones(shape=mjd[1].shape)*central_wave_length[1],
                                     np.ones(shape=mjd[2].shape)*central_wave_length[2],
                                      np.ones(shape=mjd[3].shape)*central_wave_length[3],
                                     np.ones(shape=mjd[4].shape)*central_wave_length[4],
                                     np.ones(shape=mjd[5].shape)*central_wave_length[5]])
        """
        mjd_stack = np.hstack([mjd[0],mjd[1],
                                   mjd[2]])
            
        flux_stack = np.hstack([flux[0],flux[1],
                                    flux[2]])
            
        flux_err_stack = np.hstack([flux_err[0],flux_err[1],
                                        flux_err[2]])
            
        wavelength_stack = np.hstack([np.ones(shape=mjd[0].shape)*central_wave_length[0],
                                      np.ones(shape=mjd[1].shape)*central_wave_length[1],
                                     np.ones(shape=mjd[2].shape)*central_wave_length[2]])
      
        
        
        #end

        #delet some outlines
        
        gind = np.argsort(mjd_stack)
        time = mjd_stack[gind]
        fluxes = flux_stack[gind]
        flux_errs = flux_err_stack[gind]
        filters = wavelength_stack[gind]
        """
        flux_err_mean = np.mean(flux_errs)
        flux_err_std = np.std(flux_errs)
        
        for i in range(len(time)):
            if flux_errs[i] > flux_err_mean + 3 * flux_err_std:
                np.delete(time,i)
                np.delete(fluxes,i)
                np.delete(flux_errs,i)
                np.delete(filters,i)
        """
        return (time,fluxes,flux_errs,filters)
    
    
    def fit_gps(self,concat_lc):
        #fit with 2d Guassian process 
        import george
        import numpy as np
        time = concat_lc[0]
        fluxes = concat_lc[1]
        flux_errs = concat_lc[2]
        filters = concat_lc[3]
    
        
        signal_to_noises = np.abs(fluxes) / np.sqrt(
                flux_errs ** 2 + (1e-2 * np.max(fluxes)) ** 2
            )
        index = np.argmax(signal_to_noises)
        scale = np.abs(fluxes[index])
        
        from george import kernels
        guess_length_scale=60.0
        kernel = (0.5 * scale) ** 2 * kernels.Matern32Kernel(
                [guess_length_scale ** 2, 6000 ** 2], ndim=2
            )

        kernel.freeze_parameter("k2:metric:log_M_1_1")
        x_train = np.vstack([time, filters]).T
        gp = george.GP(kernel)
        guess_parameters = gp.get_parameter_vector()

        
        gp.compute(x_train, flux_errs)
        # end

        # optimize the fittig parameters
        from scipy.optimize import minimize
        def neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.log_likelihood(fluxes)
        def grad_neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(fluxes)
        #result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
        #print(result)

        
        
        bounds = [(0, np.log(1000 ** 2))]
        bounds = [(guess_parameters[0] - 10, guess_parameters[0] + 10)] + bounds
        result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like, bounds=bounds)
        

        gp.set_parameter_vector(result.x)
        return gp
        #print("\nFinal ln-likelihood: {0:.2f}".format(gp.log_likelihood(fluxes)))
        
    def dense_light_curve(self, gp, light_curve_info, concat_lc, mjd_to_pred, num_band, lambda_shift=1, is_generation=False):
        """
        parameters: 
            gp: Gaussian Process parameter after optimization
            light_curve_infor：light curve data
            concat_lc：light curve data after concatenate
            is_generation：contain original points or not
        """
        import numpy as np
        central_wave_length = [3570.0,4767.0,6215.0,7545.0,8708.0,10040.0]
        central_wave_length = np.array(central_wave_length)
        
        time = concat_lc[0]
        fluxes = concat_lc[1]
        flux_errs = concat_lc[2]
        filters = concat_lc[3]
        
        mjd = light_curve_info[0]
        flux = light_curve_info[1]
        flux_err = light_curve_info[2]
        
        mjd_dense = []
        flux_dense = []
        flux_err_dense = []
        
        for band in range(num_band):
            if is_generation:
                mjd_add = mjd_to_pred
                x_pred = np.vstack([mjd_add,np.ones(mjd_add.shape) * central_wave_length[band]])
                flux_add, flux_err_add = gp.predict(fluxes, x_pred.T, return_var=True)
                flux_add = flux_add + np.random.normal(loc = 0,
                                                   scale = np.sqrt(flux_err_add),
                                                   size = flux_add.shape)

                mjd_dense_ = np.array(list(mjd_add))
                flux_dense_ = np.array(list(flux_add))
                flux_err_dense_ = np.array(list(np.sqrt(flux_err_add)))
                
            else:
                number_of_points = 300
                mjd_add = np.linspace(min(mjd[band]),max(mjd[band]),number_of_points-len(mjd[band]))
                x_pred = np.vstack([mjd_add,np.ones(mjd_add.shape) * central_wave_length[band]])

                flux_add, flux_err_add = gp.predict(fluxes, x_pred.T, return_var=True)
                flux_add = flux_add + np.random.normal(loc = 0,
                                                   scale = np.sqrt(flux_err_add)/2,
                                                   size = flux_add.shape)
                
                mjd_dense_ = np.array(list(mjd[band])+list(mjd_add))
                flux_dense_ = np.array(list(flux[band])+list(flux_add))
                flux_err_dense_ = np.array(list(flux_err[band])+list(np.sqrt(flux_err_add)))
                
            gind = np.argsort(mjd_dense_)
            mjd_dense_ = mjd_dense_[gind]
            flux_dense_ = flux_dense_[gind]
            flux_err_dense_ = flux_err_dense_[gind]
            mjd_dense.append(mjd_dense_)
            flux_dense.append(flux_dense_)
            flux_err_dense.append(flux_err_dense_)
            
        return (mjd_dense, flux_dense, flux_err_dense)