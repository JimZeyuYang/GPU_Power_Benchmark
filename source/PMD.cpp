#include "PMD.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <bitset>

#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <errno.h>
#include <string.h>
#include <pthread.h>

#include <sys/types.h>
#include <sys/time.h>
#include <sys/select.h>


int open_serial_port() {
    const char* devicePath = "/dev/ttyUSB0";

        // Open the serial port
    int serial_port = open(devicePath, O_RDWR | O_NOCTTY | O_NDELAY);
    if (serial_port < 0) {
        std::cerr << "Failed to open the serial port. Error: " << errno << std::endl;  
    }

    // Configure the serial port
    struct termios tty;
    if(tcgetattr(serial_port, &tty) != 0) {
        std::cerr << "Failed to get the current configuration of the serial port. Error: " << errno << std::endl;
    }

    tty.c_cflag &= ~PARENB;             // Disable parity bit
    tty.c_cflag &= ~CSTOPB;             // Set one stop bit
    tty.c_cflag &= ~CSIZE;              // Clear the data size bits
    tty.c_cflag |= CS8;                 // Set data bits to 8
    tty.c_cflag &= ~CRTSCTS;            // Disable RTS/CTS hardware flow control

    tty.c_lflag &= ~ICANON;             // Disable canonical mode
    tty.c_lflag &= ~ECHO;               // Disable echo
    tty.c_lflag &= ~ECHOE;              // Disable erasure
    tty.c_lflag &= ~ECHONL;             // Disable new-line echo
    tty.c_lflag &= ~ISIG;               // Disable interpretation of INTR, QUIT and SUSP

    tty.c_iflag &= ~(IXON | IXOFF | IXANY); // Turn off s/w flow ctrl
    tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL); 
                                        // Disable any special handling of received bytes

    tty.c_oflag &= ~OPOST;              // Prevent special interpretation of output bytes (e.g. newline chars)
    tty.c_oflag &= ~ONLCR;              // Prevent conversion of newline to carriage return/line feed

    tty.c_cc[VTIME] = 1;                // Wait for up to 1s (10 deciseconds).
    tty.c_cc[VMIN] = 1;                 // Return as soon as any data is received.

    cfsetispeed(&tty, B115200);         // Replace with the desired baud rate
    cfsetospeed(&tty, B115200);         // Replace with the desired baud rate

    if (tcsetattr(serial_port, TCSANOW, &tty) != 0) {
        std::cerr << "Failed to configure the serial port. Error: " << errno << std::endl;
        close(serial_port);
    }

    return serial_port;
}


void send_cmd(int serial_port, const unsigned char* cmd, size_t cmd_size) {
    ssize_t bytesWritten = write(serial_port, cmd, cmd_size); 
    if (bytesWritten == -1) {
        std::cerr << "Failed to write to the serial port" << std::endl;
        close(serial_port);
    }
    usleep(50000);
}


bool handshake(int serial_port) {
    // Send the handshake command
    unsigned char cmd_handshake[] = {0x00};
    send_cmd(serial_port, cmd_handshake, sizeof(cmd_handshake));

    unsigned char handshake_response [17];
    ssize_t bytesRead = read(serial_port, &handshake_response, sizeof(handshake_response));
    if (bytesRead == -1) {
        std::cerr << "Failed to read from the serial port" << std::endl;
        close(serial_port);
    }

    if (strncmp((char*)handshake_response, "ElmorLabs PMD-USB", 17) != 0) {
        std::cerr << "Handshake failed" << std::endl;
        close(serial_port);
        return false;
    } else {
        return true;
    }
}


int get_baud_rate(int serial_port) {
    struct termios tty;
    if (tcgetattr(serial_port, &tty) != 0) {
        std::cerr << "Failed to get the current serial port configuration" << std::endl;
        close(serial_port);
    }

    speed_t baud_rate_value = cfgetospeed(&tty);
    int baud;

    switch (baud_rate_value) {
        case B9600:    baud = 9600;    break;
        case B38400:   baud = 38400;   break;
        case B115200:  baud = 115200;  break;
        case B230400:  baud = 230400;  break;
        case B460800:  baud = 460800;  break;
        case B921600:  baud = 921600;  break;
        default:       baud = -1;      std::cout << "Invalid baud rate" << std::endl;
    }
    
    return baud;
}


void change_baud_rate(int serial_port, int baud) {
    // Change the baud rate on PMD
    int parity = 2;       // None
    int datawidth = 0;    // 8 bit
    int stopbits = 0;     // 1 stop bit

    unsigned char cmd_write_UART_config[] = {
        0x08,             // The command to write the UART configuration
        (unsigned char)(baud & 0xFF), (unsigned char)((baud >> 8) & 0xFF), (unsigned char)((baud >> 16) & 0xFF), (unsigned char)((baud >> 24) & 0xFF),
        (unsigned char)(parity & 0xFF), (unsigned char)((parity >> 8) & 0xFF), (unsigned char)((parity >> 16) & 0xFF), (unsigned char)((parity >> 24) & 0xFF),
        (unsigned char)(datawidth & 0xFF), (unsigned char)((datawidth >> 8) & 0xFF), (unsigned char)((datawidth >> 16) & 0xFF), (unsigned char)((datawidth >> 24) & 0xFF),
        (unsigned char)(stopbits & 0xFF), (unsigned char)((stopbits >> 8) & 0xFF), (unsigned char)((stopbits >> 16) & 0xFF), (unsigned char)((stopbits >> 24) & 0xFF),
    };

    send_cmd(serial_port, cmd_write_UART_config, sizeof(cmd_write_UART_config));

    // Change the baud rate of the host usb port
    struct termios tty;
    if (tcgetattr(serial_port, &tty) != 0) {
        std::cerr << "Failed to get the current serial port configuration" << std::endl;
        close(serial_port);
    }

    speed_t baud_rate_value;
    switch (baud) {
        case 9600:    baud_rate_value = B9600;    break;
        case 38400:   baud_rate_value = B38400;   break;
        case 115200:  baud_rate_value = B115200;  break;
        case 230400:  baud_rate_value = B230400;  break;
        case 460800:  baud_rate_value = B460800;  break;
        case 921600:  baud_rate_value = B921600;  break;
        default:      baud_rate_value = B115200;  std::cout << "Invalid baud rate, using 115200" << std::endl;       
    }

    cfsetospeed(&tty, baud_rate_value);  cfsetispeed(&tty, baud_rate_value);

    if (tcsetattr(serial_port, TCSANOW, &tty) != 0) {
        std::cerr << "Failed to set the serial port configuration" << std::endl;
        close(serial_port);
    }

    if (get_baud_rate(serial_port) != baud || !handshake(serial_port)) {
        std::cerr << "Failed to change baud rate" << std::endl;
    }
}


void config_cont_tx(int serial_port, bool enable) {
    // if (enable)  std::cout << "Enabling continuous transmission mode... ";
    // else         std::cout << "Disabling continuous transmission mode ... ";

    int ts_bytes = 2;
    unsigned char adc_channels = 0xFF;

    unsigned char cmd_write_config_cont_tx[] = {
        0x07,
        (unsigned char)(enable & 0xFF), 
        (unsigned char)(ts_bytes & 0xFF),
        adc_channels
    };    
    send_cmd(serial_port, cmd_write_config_cont_tx, sizeof(cmd_write_config_cont_tx));
}


void* logSerialPort(void* arg) {
    ThreadArgs* args = (ThreadArgs*) arg;
    int serial_port = args->serial_port;
    std::string file_path = args->file_path + "PMD_data.bin";
    

    char buffer[256];
    std::ofstream outFile(file_path, std::ios::binary);

    fd_set set;
    struct timeval timeout;

    timeout.tv_sec = 2;

    while (true) {
        FD_ZERO(&set); // clear the set
        FD_SET(serial_port, &set); // add our file descriptor to the set
        
        int rv = select(serial_port + 1, &set, NULL, NULL, &timeout);

        if (rv == -1) {
            std::cerr << "Error reading from serial port" << std::endl;
            break;
        } else if (rv == 0) {
            std::cout << "Timeout, no data received within 2 seconds" << std::endl;
            break;
        } else {
            int bytes_read = read(serial_port, &buffer, sizeof(buffer));
            std::cout << "Bytes read: " << bytes_read << std::endl;
            if (bytes_read > 0) {
                outFile.write(buffer, bytes_read);
            }
        }
        usleep(500);
    }

    outFile.close();
    pthread_exit(NULL);
}



