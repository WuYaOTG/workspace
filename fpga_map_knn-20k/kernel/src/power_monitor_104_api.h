#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <sched.h>
#include <string.h>
#include <iostream>
#include "i2c-dev.h"


/* PMBUS Commands */
#define CMD_PAGE				0x00
#define CMD_VOUT_MODE			0x20
#define CMD_READ_VOUT			0x8B
#define CMD_READ_IOUT			0x8C
#define CMD_READ_POUT			0x96
#define CMD_POWER_GOOD_OFF		0x5F
#define CMD_POWER_GOOD_ON		0x5E
#define CMD_VOUT_UV_FAULT_LIMIT	0x44
#define CMD_VOUT_UV_WARN_LIMIT	0x43
#define CMD_VOUT_MARGIN_LOW		0x26
#define CMD_VOUT_COMMAND		0x21
#define CMD_VOUT_MARGIN_HIGH	0x25
#define CMD_VOUT_OV_WARN_LIMIT	0x42
#define CMD_VOUT_OV_FAULT_LIMIT	0x40
#define CMD_VOUT_MAX			0x24

#define NUM_SENSORS 2

struct voltage_rail {
	char *name;
	unsigned char device;
	unsigned char page;
	float  voltage;
	double average_current;
	double average_power;
};


/*-------------------------------------------------------------------
     * Estructure for measured energy at one point used in
     * energy_meter_read() and energy_meter_diff()
     * */

    struct em_t
    {
        double energy[NUM_SENSORS]; // energy measured in Joules
        double current_time_stamp;
    };

/*-------------------------------------------------------------------
     * Internal structute for manage energy sampling
     * created at energy_meter_init()
     * freed at energy_meter_destroy()
     * */
    struct energy_sample
    {
        int sample_rate; // in microseconds to use in usleep()

        // cumulative energy (and final result) in Joules
	    double energy[NUM_SENSORS];

        double power[NUM_SENSORS]; // last measurement

        int destroy;  	// 1 = sampling thread must exit
        int stop;     	//  not in use

        long samples;  	// # of samples

        struct timespec start_time; // clock_gettime(CLOCK_REALTIME, ...) at starting point
        struct timespec stop_time; 	// clock_gettime(CLOCK_REALTIME, ...) at finish
        double time;       			// elapsed time in seconds (final result)

        struct timespec res[2];    // to take times between samples
        int now;

        pthread_t th_meter;    // sampling thread
        pthread_mutex_t mutex; // mutex for controling sampling thread
    };


double linear11ToFloat(unsigned char highByte, unsigned char lowByte);

float readVoltage(int iic_fd, unsigned char deviceAddress, unsigned char pageAddress);

float readCurrent(int iic_fd, unsigned char deviceAddress, unsigned char pageAddress);

float scaleVoltage(int iic_fd, unsigned char device_address, unsigned char page, float desired_voltage);

//-------------------------------------------------------------------
//- PUBLIC API ------------------------------------------------------
//-------------------------------------------------------------------

    /*------------------------------------------------------------------
      struct energy_sample * energy_meter_init(int sample_rate, int debug);

      It creates a new energy_sample structrure and internal sampling thread ready to go
      sample_rate is the sampling period in miliseconds must be < 1 second
      debug = 1 will output a file "debug_energy_meter####.txt" with samplig raw data
      Example code:
    	struct energy_sample * mysample;
    	...
    	mysample = energy_meter_init(50, 0);  // for 50 ms sampling period and no debugging
    */
    struct energy_sample * energy_meter_init(int sample_rate, int debug);

    /*-------------------------------------------------------------------
      void energy_meter_start(struct energy_sample *sample);

      It starts energy sampling thread, it also get time at starting point (stored in estructure)
      Example code:
    	energy_meter_start(mysample);
    */
    void energy_meter_start(struct energy_sample *sample);

    /*-------------------------------------------------------------------
      void energy_meter_stop(struct energy_sample *sample);

      It stops energy sampling thread, it also get time at end point (stored in estructure)
     Example code:
    	energy_meter_stop(mysample);
    	printf("A15 total energy measured= %lf Joules\n", sample->A15 );  // energy is in Joules

    */
    void energy_meter_stop(struct energy_sample *sample);

    /*-------------------------------------------------------------------
      void energy_meter_printf(struct energy_sample *sample1, FILE * fout);
      It prints energy totals and more on the file
      fout can be also stdout or stderr
     Example code:
    	energy_meter_printf(mysample, stdout);
    	...// you get somthing like:
    +--------------------+
    | POWER MEASUREMENTS |
    +--------------------+
    A7= 0.451691 J :: A15= 206.333552 J :: GPU= 0.280862 J :: Mem= 4.438422 J
    CLOCK_REALTIME = 49.533113 sec
    # of samples: 2429
    sample every (real) = 0.020392 sec
    sample every: 0.020000 sec
    */
    void energy_meter_printf(struct energy_sample *sample1, FILE * fout);

    /*-------------------------------------------------------------------
      It destroy structures, finish sampling thread and free memory
      To use always after stop
     Example code:
    	energy_meter_destroy(mysample);

     */
    void energy_meter_destroy(struct energy_sample *sample);

    /*-------------------------------------------------------------------
      void energy_meter_read(struct energy_sample *sample, struct em_t * read);
      read the current accumulated energy (Makes a new sample => readings are updated up to this moment)
     Example code:
        struct em_t read;
        ...
    	energy_meter_read(mysample, &read);
    	printf("A15 partial energy measured= %lf Joules\n", read->A15);  // energy is in Joules

    */
    void energy_meter_read(struct energy_sample *sample, struct em_t * read);

    /*-------------------------------------------------------------------
      void energy_meter_diff(struct energy_sample *sample, struct em_t * start_diff);
      read the current accumulated energy between to points (A) and (B) (see example)
      * (Makes a new sample => readings are updated up to this moment)
     Example code:
        struct em_t read;
        ...
    	energy_meter_read(mysample, &read);  //(A)
    	...
    	...
    	...
    	energy_meter_diff(mysample, &read);  //(B)
    	printf("A15 partial energy between two measurements= %lf Joules\n", read->A15);  // energy is in Joules

    */
    void energy_meter_diff(struct energy_sample *sample, struct em_t * start_diff);

    /*-------------------------------------------------------------------
      void energy_meter_read_printf(struct em_t * read_or_diff, FILE * fout);
      It prints energy in the read structure
      fout can be also stdout or stderr
     Example code:
    	energy_meter_read_printf(&read, stdout);
    	...// you get somthing like:
    POWER READ ----------------
     A7= 0.000000 J :: A15= 0.000000 J :: GPU= 0.000000 J :: Mem= 0.000000 J
    */
    void energy_meter_read_printf(struct em_t * read_or_diff, FILE *fout);

//-------------------------------------------------------------------
//-----INTERNAL USE -------------------------------------------------
    struct timespec diff(struct timespec start, struct timespec end); // internal use
    void *meter_function(void *arg); // internal use thread function
    void *meter_function_debug(void *arg); // internal use thread function
//-------------------------------------------------------------------

void read_sensors(double *power);