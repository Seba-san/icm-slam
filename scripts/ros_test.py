import roslibpy


def handler(msg):
    print('Heard talking: ' + msg['data'])

if __name__=='__main__':
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    print('Is ROS connected?', client.is_connected)
    #client.terminate()


    listener = roslibpy.Topic(client, '/chatter', 'std_msgs/String')
    listener.subscribe(handler)

    try:
        while True:
            pass
    except KeyboardInterrupt:
        client.terminate()
