"""Lightweight protocol-field dissector used as our substitute for Wireshark
(tshark/pyshark are not installed in this environment; see DATA_AUDIT.md).
Produces compact 'Layer:field=value;...' text in the spirit of the source
paper's custom-vocabulary field tokens (Table II: ETH_Layer, IP_Layer, ...).
"""
from scapy.layers.l2 import Ether, ARP
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.contrib.modbus import ModbusADURequest, ModbusADUResponse

try:
    from scapy.contrib.modbus import (
        ModbusPDU01ReadCoilsRequest, ModbusPDU03ReadHoldingRegistersRequest,
        ModbusPDU05WriteSingleCoilRequest, ModbusPDU06WriteSingleRegisterRequest,
        ModbusPDU0FWriteMultipleCoilsRequest, ModbusPDU10WriteMultipleRegistersRequest,
    )
    _MODBUS_PDUS = True
except Exception:
    _MODBUS_PDUS = False


def dissect_to_text(pkt) -> str:
    """Return a compact structured-text dissection of one packet, mirroring the
    kind of target text a Wireshark dissector would emit, using only field
    names/values (no ASCII-art)."""
    parts = []
    if pkt.haslayer(Ether):
        e = pkt[Ether]
        parts.append(f"ETH_Layer:src={e.src},dst={e.dst},type={e.type}")
    if pkt.haslayer(ARP):
        a = pkt[ARP]
        parts.append(f"ARP_Layer:op={a.op},psrc={a.psrc},pdst={a.pdst},hwsrc={a.hwsrc},hwdst={a.hwdst}")
    if pkt.haslayer(IP):
        ip = pkt[IP]
        parts.append(f"IP_Layer:src={ip.src},dst={ip.dst},proto={ip.proto},ttl={ip.ttl},len={ip.len}")
    if pkt.haslayer(TCP):
        t = pkt[TCP]
        parts.append(f"TCP_Layer:sport={t.sport},dport={t.dport},flags={t.flags},seq={t.seq},ack={t.ack},window={t.window}")
    if pkt.haslayer(UDP):
        u = pkt[UDP]
        parts.append(f"UDP_Layer:sport={u.sport},dport={u.dport},len={u.len}")
    if pkt.haslayer(ICMP):
        ic = pkt[ICMP]
        parts.append(f"ICMP_Layer:type={ic.type},code={ic.code},id={getattr(ic, 'id', '')},seq={getattr(ic, 'seq', '')}")
    if pkt.haslayer(ModbusADURequest) or pkt.haslayer(ModbusADUResponse):
        mb = pkt[ModbusADURequest] if pkt.haslayer(ModbusADURequest) else pkt[ModbusADUResponse]
        fc = getattr(mb.payload, "funcCode", None)
        parts.append(f"Modbus_Layer:transId={mb.transId},protoId={mb.protoId},unitId={mb.unitId},funcCode={fc}")
        pdu = mb.payload
        pdu_fields = []
        for fname in ("startAddr", "quantity", "outputAddr", "outputValue", "registerAddr", "registerValue", "byteCount"):
            if hasattr(pdu, fname):
                pdu_fields.append(f"{fname}={getattr(pdu, fname)}")
        if pdu_fields:
            parts.append("Modbus_PDU:" + ",".join(pdu_fields))
    if not parts:
        parts.append(f"Raw_Layer:len={len(pkt)}")
    return ";".join(parts)


def raw_bytes(pkt) -> bytes:
    return bytes(pkt)
