#include "power_monitor_104_api.h"

struct voltage_rail zcu104_rails[] = {
	{
		name			: "VCCINT   ",
		device			: 0x43,
		page			: 0x00,
		voltage			: 0.0,
		average_current	: 0.0,
		average_power	: 0.0
	},
	{
		name			: "VCC1V8   ",
		device			: 0x43,
		page			: 0x01,
		voltage			: 0.0,
		average_current	: 0.0,
		average_power	: 0.0
	}
};

double linear11ToFloat(unsigned char highByte, unsigned char lowByte) {
	unsigned short combinedWord;
	signed char exponent;
	signed short mantissa;
	double current;

	combinedWord = highByte;
	combinedWord <<= 8;
	combinedWord += lowByte;

	exponent = combinedWord >> 11;
	mantissa = combinedWord & 0x7ff;

	/* Sign extend the exponent and the mantissa */
	if(exponent > 0x0f) {
		exponent |= 0xe0;
	}
	if(mantissa > 0x03ff) {
		mantissa |= 0xf800;
	}

	current = mantissa * pow(2.0, exponent);

	return (float)current;
}

float readVoltage(int iic_fd, unsigned char deviceAddress, unsigned char pageAddress) {
	float voltage;
	int status;
	char VOUT_MODE;
	if (ioctl(iic_fd, I2C_SLAVE_FORCE, deviceAddress) < 0) {
		printf("ERROR: Unable to set I2C slave address 0x%02X\n", deviceAddress);
		exit(1);
	}

	status = i2c_smbus_write_byte_data(iic_fd, CMD_PAGE, pageAddress);
	if (status < 0) {
		printf("ERROR: Unable to write page address to I2C slave at 0x%02X: %d\n", deviceAddress, status);
		exit(1);
	}

	VOUT_MODE = i2c_smbus_read_byte_data(iic_fd, CMD_VOUT_MODE);
	if (status < 0) {
		printf("ERROR: Unable to read VOUT_MODE address to I2C slave at 0x%02X: %d\n", deviceAddress, status);
		exit(1);
	}

	/* Read in the voltage value */
	status = i2c_smbus_read_word_data(iic_fd, CMD_READ_VOUT);
	if(status < 0) {
		printf("ERROR: Unable to read VOUT on I2C slave at 0x%02X: %d\n", deviceAddress, status);
		exit(1);
	}

	voltage = status*pow(2.0, VOUT_MODE-32);
	return voltage;
}

float readCurrent(int iic_fd, unsigned char deviceAddress, unsigned char pageAddress) {
	double current;
	int status;

	if (ioctl(iic_fd, I2C_SLAVE_FORCE, deviceAddress) < 0) {
		printf("ERROR: Unable to set I2C slave address 0x%02X\n", deviceAddress);
		exit(1);
	}

	status = i2c_smbus_write_byte_data(iic_fd, CMD_PAGE, pageAddress);
	if (status < 0) {
		printf("ERROR: Unable to write page address to I2C slave at 0x%02X: %d\n", deviceAddress, status);
		exit(1);
	}

	status = i2c_smbus_read_word_data(iic_fd, CMD_READ_IOUT);
	if(status < 0) {
		printf("ERROR: Unable to read IOUT on I2C slave at 0x%02X: %d\n", deviceAddress, status);
		exit(1);
	}

	current = linear11ToFloat((unsigned char)((status >> 8) & 0xff), (unsigned char)(status & 0xff));
	return current;
}

float scaleVoltage(int iic_fd, unsigned char device_address, unsigned char page, float desired_voltage) {

	int status;
	float sample_voltage;
	int k = 0;
	char VOUT_MODE;
	__u16 VOUT_MAX,VOUT_OV_FAULT_LIMIT,VOUT_OV_WARN_LIMIT,VOUT_MARGIN_HIGH,VOUT_COMMAND,
		VOUT_MARGIN_LOW,VOUT_UV_WARN_LIMIT,VOUT_UV_FAULT_LIMIT,POWER_GOOD_ON,POWER_GOOD_OFF;

	float voltage_min=0.20;
	float voltage_max=1.10;

	sample_voltage = readVoltage(iic_fd, device_address, page);
	VOUT_MODE = i2c_smbus_read_byte_data(iic_fd, CMD_VOUT_MODE);

	if (desired_voltage <= voltage_max && desired_voltage >= voltage_min) {

		VOUT_COMMAND = (__u16)(desired_voltage/pow(2.0, VOUT_MODE-32));

		VOUT_MAX =(__u16)(VOUT_COMMAND*1.6);
		VOUT_OV_FAULT_LIMIT =(__u16)(VOUT_COMMAND*1.1);
		VOUT_OV_WARN_LIMIT=(__u16)(VOUT_COMMAND*1.075);
		VOUT_MARGIN_HIGH=(__u16)(VOUT_COMMAND*1.05);
		VOUT_MARGIN_LOW=(__u16)(VOUT_COMMAND*0.95);
		VOUT_UV_WARN_LIMIT=(__u16)(VOUT_COMMAND*0.925);
		VOUT_UV_FAULT_LIMIT=(__u16)(VOUT_COMMAND*0.9);
		POWER_GOOD_ON=(__u16)(VOUT_COMMAND*0.925);
		POWER_GOOD_OFF=(__u16)(VOUT_COMMAND*0.9);

		if (ioctl(iic_fd, I2C_SLAVE_FORCE, device_address) < 0) {
			printf("scale_voltage: ERROR 07: Unable to set I2C slave address 0x%02X\n", device_address);
			exit(1);
		}

		status = i2c_smbus_write_byte_data(iic_fd, CMD_PAGE, page);
		if (status < 0) {
			printf("scale_voltage: ERROR 08: Unable to write page address to I2C slave at 0x%02X: %d\n", device_address, status);
			exit(1);
		}

		if (sample_voltage > desired_voltage)//scaling voltage down
		{

			status = i2c_smbus_write_word_data(iic_fd,CMD_POWER_GOOD_OFF,POWER_GOOD_OFF);
			if (status < 0) {
				printf("ERROR 09: Unable to write page address to I2C slave at 0x%02X: %d\n", device_address, status);
				exit(1);
			}

			status = i2c_smbus_write_word_data(iic_fd,CMD_POWER_GOOD_ON,POWER_GOOD_ON);
			if (status < 0) {
				printf("ERROR 13: Unable to write page address to I2C slave at 0x%02X: %d\n", device_address, status);
				exit(1);
			}

			status = i2c_smbus_write_word_data(iic_fd,CMD_VOUT_UV_FAULT_LIMIT,VOUT_UV_FAULT_LIMIT);
			if (status < 0) {
				printf("ERROR 11: Unable to write page address to I2C slave at 0x%02X: %d\n", device_address, status);
				exit(1);
			}

			status = i2c_smbus_write_word_data(iic_fd,CMD_VOUT_UV_WARN_LIMIT,VOUT_UV_WARN_LIMIT);
			if (status < 0) {
				printf("ERROR 12: Unable to write page address to I2C slave at 0x%02X: %d\n", device_address, status);
				exit(1);
			}

			status = i2c_smbus_write_word_data(iic_fd,CMD_VOUT_MARGIN_LOW,VOUT_MARGIN_LOW);
			if (status < 0) {
				printf("ERROR 14: Unable to write page address to I2C slave at 0x%02X: %d\n", device_address, status);
				exit(1);
			}

			status = i2c_smbus_write_word_data(iic_fd,CMD_VOUT_COMMAND,VOUT_COMMAND);
			if (status < 0) {
				printf("ERROR 15: Unable to write page address to I2C slave at 0x%02X: %d\n", device_address, status);
				exit(1);
			}

			status = i2c_smbus_write_word_data(iic_fd,CMD_VOUT_MARGIN_HIGH,VOUT_MARGIN_HIGH);
			if (status < 0) {
				printf("ERROR 16: Unable to write page address to I2C slave at 0x%02X: %d\n", device_address, status);
				exit(1);
			}

			status = i2c_smbus_write_word_data(iic_fd,CMD_VOUT_OV_WARN_LIMIT,VOUT_OV_WARN_LIMIT);
			if (status < 0) {
				printf("ERROR 17: Unable to write page address to I2C slave at 0x%02X: %d\n", device_address, status);
				exit(1);
			}

			status = i2c_smbus_write_word_data(iic_fd,CMD_VOUT_OV_FAULT_LIMIT,VOUT_OV_FAULT_LIMIT);
			if (status < 0) {
				printf("ERROR 18: Unable to write page address to I2C slave at 0x%02X: %d\n", device_address, status);
				exit(1);
			}

			status = i2c_smbus_write_word_data(iic_fd,CMD_VOUT_MAX,VOUT_MAX);
			if (status < 0) {
				printf("ERROR 19: Unable to write page address to I2C slave at 0x%02X: %d\n", device_address, status);
				exit(1);
			}

		} 
		else //scaling voltage up
		{

			status = i2c_smbus_write_word_data(iic_fd,CMD_VOUT_OV_FAULT_LIMIT,VOUT_OV_FAULT_LIMIT);
			if (status < 0) {
				printf("ERROR 21: Unable to write page address to I2C slave at 0x%02X: %d\n", device_address, status);
				exit(1);
			}

			status = i2c_smbus_write_word_data(iic_fd,CMD_VOUT_OV_WARN_LIMIT,VOUT_OV_WARN_LIMIT);
			if (status < 0) {
				printf("ERROR 22: Unable to write page address to I2C slave at 0x%02X: %d\n", device_address, status);
				exit(1);
			}

			status = i2c_smbus_write_word_data(iic_fd,CMD_VOUT_MARGIN_HIGH,VOUT_MARGIN_HIGH);
			if (status < 0) {
				printf("ERROR 23: Unable to write page address to I2C slave at 0x%02X: %d\n", device_address, status);
				exit(1);
			}

			status = i2c_smbus_write_word_data(iic_fd,CMD_VOUT_COMMAND,VOUT_COMMAND);
			if (status < 0) {
				printf("ERROR 24: Unable to write page address to I2C slave at 0x%02X: %d\n", device_address, status);
				exit(1);
			}

			status = i2c_smbus_write_word_data(iic_fd,CMD_VOUT_MARGIN_LOW,VOUT_MARGIN_LOW);
			if (status < 0) {
				printf("ERROR 25: Unable to write page address to I2C slave at 0x%02X: %d\n", device_address, status);
				exit(1);
			}

			status = i2c_smbus_write_word_data(iic_fd,CMD_VOUT_UV_WARN_LIMIT,VOUT_UV_WARN_LIMIT);
			if (status < 0) {
				printf("ERROR 27: Unable to write page address to I2C slave at 0x%02X: %d\n", device_address, status);
				exit(1);
			}

			status = i2c_smbus_write_word_data(iic_fd,CMD_VOUT_UV_FAULT_LIMIT,VOUT_UV_FAULT_LIMIT);
			if (status < 0) {
				printf("ERROR 29: Unable to write page address to I2C slave at 0x%02X: %d\n", device_address, status);
				exit(1);
			}

			status = i2c_smbus_write_word_data(iic_fd,CMD_POWER_GOOD_ON,POWER_GOOD_ON);
			if (status < 0) {
				printf("ERROR 26: Unable to write page address to I2C slave at 0x%02X: %d\n", device_address, status);
				exit(1);
			}

			status = i2c_smbus_write_word_data(iic_fd,CMD_POWER_GOOD_OFF,POWER_GOOD_OFF);
			if (status < 0) {
				printf("ERROR 30: Unable to write page address to I2C slave at 0x%02X: %d\n", device_address, status);
				exit(1);
			}
		}
	}
}


void read_sensors(double *power)
{
    int iic_fd;
	iic_fd = open("/dev/i2c-4", O_RDWR);
	if (iic_fd < 0) {
		printf("ERROR: Unable to open /dev/i2c-4 for PMBus access: %d\n", iic_fd);
		exit(1);
	}

    for(int i = 0; i < NUM_SENSORS; i++)
    {
        double voltage = readVoltage(iic_fd, zcu104_rails[i].device, zcu104_rails[i].page);
        double current = readCurrent(iic_fd, zcu104_rails[i].device, zcu104_rails[i].page);
        double measure_power = voltage * current * 1000;    // real_current = current * 1000;
        power[i] = measure_power;
    }
	close(iic_fd);
}

struct energy_sample * energy_meter_init(int sample_rate, int debug) // sample rate in miliseconds
{
	struct energy_sample * sample;
	int i;
	sample=(struct energy_sample *) malloc(sizeof(struct energy_sample ));
		
	sample->sample_rate=sample_rate*1000; // in microseconds to use in usleep()
	for(i=0; i<NUM_SENSORS; i++) sample->energy[i]=0.0;
	sample->destroy=0;
	sample->stop=1;
	sample->samples=0;
	
	pthread_mutex_init(&(sample->mutex),NULL);
	pthread_mutex_lock(&(sample->mutex));
	if (debug)
	pthread_create(&(sample->th_meter), NULL, meter_function_debug , (void *)sample);	//thread
	else
	pthread_create(&(sample->th_meter), NULL, meter_function , (void *)sample);	//thread 
	
	return(sample);
}

//-------------------------------------------------------------------
void energy_meter_start(struct energy_sample *sample)
{
	clock_gettime(CLOCK_REALTIME, &(sample->start_time));
	
	pthread_mutex_unlock(&(sample->mutex)); //start energy sampling
}

//-------------------------------------------------------------------
void energy_meter_stop(struct energy_sample *sample)
{
	struct timespec res;
	double secs;
	struct timespec dif;
	int i;
   
	pthread_mutex_lock(&(sample->mutex));  // stop energy sampling
	clock_gettime(CLOCK_REALTIME, &(sample->stop_time));
	res=diff(sample->start_time, sample->stop_time);
	
	sample->time=(double)res.tv_sec+ (double)res.tv_nsec/1000000000.0;
	
	
	//read_sensors(sample1->a7W, sample1->a15W, sample1->gpuW, sample1->memW);
    //sample->now=!sample->now;
	// get time now**********************************
	//clock_gettime(CLOCK_REALTIME, &dif );
	// get time interval    !!! only nanoseconds, sampling rate must be below 1 second
	dif.tv_nsec=sample->stop_time.tv_nsec - sample->res[sample->now].tv_nsec;
	if(	dif.tv_nsec <0)	dif.tv_nsec += 1000000000;
	// claculate energy until now **************************************
	secs= dif.tv_nsec/1000000000.0; // move to seconds
	for (i=0; i<NUM_SENSORS; i++) 
	sample->energy[i] += sample->power[i] * secs;
	
}
//-------------------------------------------------------------------

void energy_meter_destroy(struct energy_sample *sample) // always after stop
{
	sample->destroy=1;
	pthread_mutex_unlock(&(sample->mutex));  
	pthread_join(sample->th_meter,NULL);
	pthread_mutex_destroy(&(sample->mutex));
	free(sample);
}
//-------------------------------------------------------------------
void energy_meter_printf(struct energy_sample *sample1, FILE * fout)
{
	struct timespec res;
	int i;
	res=diff(sample1->start_time, sample1->stop_time);
	double total_time = (double)res.tv_sec+ (double)res.tv_nsec/1000000000.0;
	
	fprintf(fout,"+--------------------+\n");
	fprintf(fout,"| POWER MEASUREMENTS |\n");
	fprintf(fout,"+--------------------+\n");
	
    double total_energy = 0.0;
	for (i=0; i<NUM_SENSORS; i++) {
        fprintf(fout,"%14s = %8.3lf mJ ",zcu104_rails[i].name,sample1->energy[i]);
        fprintf(fout,"\n");
        total_energy += sample1->energy[i];
    }

	fprintf(fout,"TOTAL ENERGY = %lf mJ, AVERAGE POWER = %lf mW,  CLOCK_REALTIME = %lf sec\n",total_energy,  total_energy/total_time, total_time);
	fprintf(fout,"\n");
	fprintf(fout,"# of samples: %ld\n", sample1->samples);
	fprintf(fout,"sample every (real) = %lf sec\n",((double)res.tv_sec+ (double)res.tv_nsec/1000000000.0)/sample1->samples);	
	fprintf(fout,"sample every: %lf sec\n",(double)sample1->sample_rate/1000000);	
	
}
//-------------------------------------------------------------------

void energy_meter_read(struct energy_sample *sample, struct em_t * out)
{
	double secs;
	struct timespec dif;
	int i;
   
	// mutex 
	pthread_mutex_lock(&(sample->mutex));
	
		dif=diff(sample->start_time, sample->res[sample->now]);
		//fprintf(debugf,"%lf  ", (double)dif.tv_sec+ (double)dif.tv_nsec/1000000000.0);
		out->current_time_stamp = ((double)dif.tv_sec+ (double)dif.tv_nsec/1000000000.0);

		sample->now=!sample->now;
		// get time now**********************************
		clock_gettime(CLOCK_REALTIME, sample->res+sample->now );
		read_sensors(sample->power);
		// get time interval    !!! only nanoseconds, sampling rate must be below 1 second
		dif.tv_nsec=sample->res[sample->now].tv_nsec-sample->res[!sample->now].tv_nsec;
		if(	dif.tv_nsec <0)	dif.tv_nsec += 1000000000;
		
		
		// claculate energy  **************************************
		secs= dif.tv_nsec/1000000000.0; // move to seconds
		for (i=0; i<NUM_SENSORS; i++) 
		{
		    sample->energy[i] += sample->power[i] * secs;
	        out->energy[i]=  sample->energy[i]/1000.0;
		}
        
				
		sample->samples++;
	
	pthread_mutex_unlock(&(sample->mutex));
	
	
	//
	
}
//-------------------------------------------------------------------
void energy_meter_diff(struct energy_sample *sample, struct em_t * diff)
{
	double secs;
	struct timespec dif;
	int i;
   
	// mutex 
	pthread_mutex_lock(&(sample->mutex));
	
		sample->now=!sample->now;
		// get time now**********************************
		clock_gettime(CLOCK_REALTIME, sample->res+sample->now );
		read_sensors(sample->power);
		// get time interval    !!! only nanoseconds, sampling rate must be below 1 second
		dif.tv_nsec=sample->res[sample->now].tv_nsec-sample->res[!sample->now].tv_nsec;
		if(	dif.tv_nsec <0)	dif.tv_nsec += 1000000000;
		
		
		// claculate energy  **************************************
		secs= dif.tv_nsec/1000000000.0; // move to seconds
        
		for (i=0; i<NUM_SENSORS; i++) 
		{
		sample->energy[i] += sample->power[i] * secs;
	        diff->energy[i]=  sample->energy[i]/1000.0 - diff->energy[i];
		}
				
		sample->samples++;
	
	pthread_mutex_unlock(&(sample->mutex));

}
//-------------------------------------------------------------------
void energy_meter_read_printf(struct em_t * sample1, FILE *fout)
{
	int i;
    double total_energy = 0.0;
	fprintf(fout,"POWER READ --------------------------------------------\n");
    for (i=0; i<NUM_SENSORS; i++) {
        fprintf(fout,"%14s = %8.3lf mJ ",zcu104_rails[i].name,sample1->energy[i]);
        fprintf(fout,"\n");
        total_energy += sample1->energy[i];
    }
    fprintf(fout,"\n");
    fprintf(fout,"TOTAL ENERGY= %lf mJ\n", total_energy );
	fprintf(fout,"\n");
}
//-------------------------------------------------------------------


void *meter_function(void *arg)
{
	struct energy_sample *sample=(struct energy_sample *) arg;
	
	char buf[256];
	//int fa7, fa15,fgpu,fmem;
	struct timespec dif;
	double secs;
	int i;
	
    sample->now=0;
    // first sample
 	pthread_mutex_lock(&(sample->mutex));
 	clock_gettime(CLOCK_REALTIME, sample->res);
 	read_sensors(sample->power);
    pthread_mutex_unlock(&(sample->mutex));
	
	usleep(sample->sample_rate);

	while(1)  // sampling on course
	{
		pthread_mutex_lock(&(sample->mutex));
		if(sample->destroy)
		{
			pthread_mutex_unlock(&(sample->mutex));
			pthread_exit(NULL);
		}
		sample->now=!sample->now;
		// get time now**********************************
		clock_gettime(CLOCK_REALTIME, sample->res+sample->now );
		read_sensors(sample->power);
		// get time interval    !!! only nanoseconds, sampling rate must be below 1 second
		dif.tv_nsec=sample->res[sample->now].tv_nsec-sample->res[!sample->now].tv_nsec;
		if(	dif.tv_nsec <0)	dif.tv_nsec += 1000000000;
		
		// claculate energy  **************************************
		secs= dif.tv_nsec/1000000000.0; // move to seconds
        
//		sample->A7  += sample->a7W * secs ; // Watt*sec=Joules
		for (i=0; i<NUM_SENSORS; i++) 
		    sample->energy[i] += sample->power[i] * secs;
		
	
		sample->samples++;
		// DEBUG
		// fprintf(stdout,"a7= %lf W : a15= %lf W : gpu= %lf W \n",a7W,a15W,gpuW);
		// fprintf(stdout,"CLOCK_REALTIME = %lld sec, %ld nsec\n",(long long) dif.tv_sec, (long)dif.tv_nsec);	
	
		
		
		pthread_mutex_unlock(&(sample->mutex));
		
		usleep(sample->sample_rate);
	}
	
}


//-------------------------------------------------------------------

void *meter_function_debug(void *arg)
{
	struct energy_sample *sample=(struct energy_sample *) arg;
	struct timespec dif;
	char buf[256];
	FILE *debugf;
	int fa7, fa15,fgpu, fmem;
	//double a7W=0.0, a15W=0.0, gpuW=0.0, memW=0.0;
	double secs;
    int c1,c2,c3,c4,cGPU;
    int i;
    char fn[256];
    sprintf(fn,"DEBUG_energy_meter.txt");
    debugf=fopen(fn,"w");
    fprintf(debugf,"#;sample;time;sensors\n");	
    // first sample
    sample->now=0;
 	pthread_mutex_lock(&(sample->mutex));
 	clock_gettime(CLOCK_REALTIME, sample->res);
 	read_sensors(sample->power);
	
    pthread_mutex_unlock(&(sample->mutex));
	
	usleep(sample->sample_rate);

	while(1)  // sampling on course
	{
		pthread_mutex_lock(&(sample->mutex));
		if(sample->destroy)
		{
			pthread_mutex_unlock(&(sample->mutex));
			fclose(debugf);
			pthread_exit(NULL);
		}
		sample->now=!sample->now;
		// get time now**********************************
		clock_gettime(CLOCK_REALTIME, sample->res+sample->now );
		// get time interval    !!! only nanoseconds, sampling rate must be below 1 second
		dif.tv_nsec=sample->res[sample->now].tv_nsec-sample->res[!sample->now].tv_nsec;
		if(	dif.tv_nsec <0)	dif.tv_nsec += 1000000000;
		
		read_sensors(sample->power);
		
		// claculate energy  **************************************
		secs= dif.tv_nsec/1000000000.0; // move to seconds
        
		for (i=0; i<NUM_SENSORS; i++) 
		sample->energy[i] += sample->power[i] * secs;

		// read sensors ********************************* 
		
		sample->samples++;
		// DEBUG
		fprintf(debugf,"%ld ", sample->samples);
		
		//fprintf(debugf,"%ld;", (long)dif.tv_nsec);
		
		dif=diff(sample->start_time, sample->res[sample->now]); 
	
	//sample->time=(double)res.tv_sec+ (double)res.tv_nsec/1000000000.0;

		fprintf(debugf,"%lf  ", (double)dif.tv_sec+ (double)dif.tv_nsec/1000000000.0);

        for(i=0; i<NUM_SENSORS; i++)
        	fprintf(debugf,"%lf ",sample->power[i]);

		fprintf(debugf,"\n");
		//if(sample->samples<20) printf("%lf;",memW);
		
		//read_sensors(sample->a7W, sample->a15W, sample->gpuW, sample->memW);
		
		pthread_mutex_unlock(&(sample->mutex));
		
		usleep(sample->sample_rate);
	}
	
}


//-------------------------------------------------------------------

struct timespec diff(struct timespec start, struct timespec end)
{
	struct timespec temp;
	if ((end.tv_nsec-start.tv_nsec)<0) {
		temp.tv_sec = end.tv_sec-start.tv_sec-1;
		temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	} else {
		temp.tv_sec = end.tv_sec-start.tv_sec;
		temp.tv_nsec = end.tv_nsec-start.tv_nsec;
	}
	return temp;
}

//-------------------------------------------------------------------
