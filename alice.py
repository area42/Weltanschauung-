import logging
import asyncio
import argparse
import Weltantschauung
import os
import pickle

from hbmqtt.client import MQTTClient, ClientException
from hbmqtt.mqtt.constants import QOS_1, QOS_2


logger = logging.getLogger(__name__)

def isNone(arg):
    return arg.__str__() == 'None'

@asyncio.coroutine
def listen_and_learn(myname,mywv,mypartner,convo_starter=False):
    C = MQTTClient()

    m2y = "%s/%s" % (myname,mypartner)
    y2m = "%s/%s" % (mypartner,myname)

    yield from C.connect('mqtt://localhost/')
    yield from C.subscribe([
        (y2m, QOS_1),
    ])
    logger.info("Subscribed to %s" % (y2m))
    try:
        if convo_starter:
            s = mywv.random_thought()
            convo_starter = False
        else:
            s = None

        for i in range(1, 500):
            if not isNone(s):
                yield from C.publish(m2y, pickle.dumps(s))
            message = yield from C.deliver_message()
            packet = message.publish_packet
            r = pickle.loads(packet.payload.data)
            if not isNone(s):
                l = mywv.listenAndLearn(s,r)
            else:
                l = "N/A"

            print("%d: %s => %s (loss %-25s)" % (i, packet.variable_header.topic_name, r.__str__() ,l.__str__()))
            s = mywv.reply(r)
            print("%d: %s => %s " % (i, m2y, s.__str__()))

        yield from C.unsubscribe(["%s/%s" % (mypartner,myname)])
        logger.info("UnSubscribed from %s/%s" % (mypartner,myname))
        yield from C.disconnect()

    except ClientException as ce:
        logger.error("Client exception: %s" % ce)


if __name__ == '__main__':

    formatter = "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=formatter)

    parser = argparse.ArgumentParser(
            description = "Starts a conversation between myself and give bot",
            epilog = "As an alternative to the commandline, params can be placed in a file, one per line, and specified on the commandline like '%(prog)s @params.conf'.",
                fromfile_prefix_chars = '@' )
    parser.add_argument("partner",
                      help = "partner to talk to",
                      metavar = "partner")
    parser.add_argument("-i",
                      "--initiate",
                      help="I will initiate conversation",
                      action="store_true")
    parser.add_argument("-n",
                        "--name",
                        default=os.path.splitext(os.path.basename(__file__))[0],
                        help="set my name"
                        )
    args = parser.parse_args()

    # Get my name

    myname =args.name
    mypartner = args.partner
    mylearning_rate = 1e-4
    logger.info("Talking as %s to %s" % (myname,mypartner))

    mywv = Weltantschauung.Weltantschauung(myname,mylearning_rate)


    asyncio.get_event_loop().run_until_complete(listen_and_learn(myname,mywv,mypartner,args.initiate))
