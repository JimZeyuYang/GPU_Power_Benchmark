#ifndef PDM_H
#define PDM_H

#include <cstddef>
#include <string>

int open_serial_port();
void send_cmd(int serial_port, const unsigned char* cmd, size_t cmd_size);
bool handshake(int serial_port);
int get_baud_rate(int serial_port);
void change_baud_rate(int serial_port, int baud);
void config_cont_tx(int serial_port, bool enable);
void* logSerialPort(void* arg);

struct ThreadArgs {
    int serial_port;
    std::string file_path;
};

#endif // PDM_H