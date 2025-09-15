/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useActionState } from 'react';

import { Button, Flex, Item, Picker, Text, TextField } from '@geti/ui';

const protocols = ['RTSP', 'HTTP', 'HTTPS', 'ONVIF', 'SMTP', 'TCP'] as const;
type Protocol = (typeof protocols)[number];

export const IPCameraForm = () => {
    const handleSubmit = async (_state: string | null, formData: FormData) => {
        const ip = formData.get('ipField') as string;
        const port = formData.get('portField') as string;
        const streamPath = formData.get('streamPathField') as string;
        const protocol = formData.get('protocolField') as Protocol;

        if (!ip || !port || !streamPath || !protocol) {
            return 'All fields are required.';
        }

        await new Promise((res) => setTimeout(res, 1000));

        return `Saved: ${protocol}://${ip}:${port}/${streamPath}`;
    };
    const [stateMessage, formAction, isPending] = useActionState(handleSubmit, null);

    return (
        <form action={formAction}>
            <Flex direction='row' gap='size-200'>
                <TextField label='IP Address' name='ipField' />
                <TextField label='Port' name='portField' />
            </Flex>
            <TextField label='Stream Path' name='streamPathField' />
            <Picker label='Protocol' name='protocolField' items={protocols} defaultSelectedKey={protocols[0]}>
                {protocols.map((protocolKey) => (
                    <Item key={protocolKey}>{protocolKey}</Item>
                ))}
            </Picker>
            <Flex marginTop='size-200'>
                <Button type='submit' width={'size-700'} isDisabled={isPending}>
                    Apply
                </Button>
            </Flex>
            {stateMessage && <Text marginTop='size-200'>{stateMessage}</Text>}
        </form>
    );
};
